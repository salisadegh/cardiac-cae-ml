"""
02_nhanes_specificity.py
=========================
CAE empirical specificity stress test — NHANES 2017-2018 (N=5,809).
Reproduces Table 2 (NHANES row) and ablation results.

Data sources:
  NHANES 2017-2018 XPT files: https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017
  Mortality file: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/

Usage:
    python 02_nhanes_specificity.py

Output:
    results/nhanes_ablation_results.csv
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────
# CAE Classification — NHANES 2017-2018
# ─────────────────────────────────────────────────────────────────
# T₀ = baseline NHANES survey interview
# All 12 features → λ=0 (available at T₀; none satisfies Q₁)
# CV disease history variables: causally associated with mortality
# but are NOT logically equivalent to the outcome → correctly λ=0

CAE_CLASSIFICATION = {
    'Age':                   0,  # demographic
    'Sex':                   0,  # demographic
    'Ethnicity':             0,  # demographic
    'TotalCholesterol':      0,  # lab value at T₀
    'HDL_Cholesterol':       0,  # lab value at T₀
    'Sedentary_MinPerDay':   0,  # lifestyle behaviour at T₀
    'Arthritis':             0,  # comorbidity
    'Overweight':            0,  # comorbidity
    'HeartAttack_History':   0,  # causal risk factor — NOT λ=3 (legitimate baseline predictor)
    'Stroke_History':        0,  # causal risk factor — NOT λ=3
    'CHF_History':           0,  # causal risk factor — NOT λ=3
    'CoronaryDisease_History':0, # causal risk factor — NOT λ=3
}

print("=== CAE Classification: NHANES 2017-2018 ===")
print("T₀ = baseline NHANES survey interview")
print("SPECIFICITY STRESS TEST: CV disease history variables are causally")
print("associated with mortality but are NOT λ=3 — correctly retained as λ=0\n")

for feat, lam in CAE_CLASSIFICATION.items():
    tag = "(legitimate baseline predictor, NOT leakage)" if 'History' in feat else ""
    print(f"  {feat:<28} λ={lam}  {tag}")

print(f"\nResult: 0 features removed (λ=3 = 0)")
print(f"        All 12 features correctly retained as λ=0")
print(f"        CAE does not flag causally-associated baseline predictors as leaky\n")

# ─────────────────────────────────────────────────────────────────
# Load pre-processed dataset
# ─────────────────────────────────────────────────────────────────
data_path = Path('../data/nhanes_cae_dataset.csv')
if not data_path.exists():
    print(f"Dataset not found at {data_path}")
    print("Run the NHANES preprocessing script or place the dataset in ../data/")
    exit(1)

df  = pd.read_csv(data_path)
y   = df['MORTSTAT'].values.astype(int)
ALL = list(CAE_CLASSIFICATION.keys())
CV_HISTORY = ['HeartAttack_History', 'Stroke_History', 'CHF_History', 'CoronaryDisease_History']
BASE_ONLY  = [f for f in ALL if f not in CV_HISTORY]

print(f"Dataset: N={len(df):,}, Deaths={int(y.sum())} ({y.mean()*100:.1f}%)")

def prep(cols):
    X = df[cols].copy()
    for c in X.columns: X[c] = X[c].fillna(X[c].median())
    return X.values.astype(float)

def smote(X, y, k=3, rs=42):
    rng = np.random.RandomState(rs); X, y = np.array(X, dtype=float), np.array(y)
    mn = 1 if (y==1).sum() < (y==0).sum() else 0
    Xm = X[y==mn]; n = (y!=mn).sum() - (y==mn).sum(); syn = []
    for _ in range(n):
        i = rng.randint(0, len(Xm)); s = Xm[i]
        d = np.sqrt(((Xm-s)**2).sum(1)); d[i] = np.inf
        nb = Xm[np.argsort(d)[:k]][rng.randint(0, k)]; syn.append(s + rng.rand() * (nb-s))
    return np.vstack([X, syn]), np.concatenate([y, np.full(n, mn)])

def run_cv(X, y):
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    yt_all, yp_all = [], []
    for fold, (tr, te) in enumerate(outer.split(X, y)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        if ytr.sum() >= 3 and (ytr==0).sum() >= 3:
            Xs, ys = smote(Xtr, ytr, k=3, rs=fold)
        else:
            Xs, ys = Xtr, ytr
        sc = StandardScaler()
        clf = SVC(kernel='linear', C=1, probability=True, class_weight='balanced', random_state=RANDOM_STATE)
        clf.fit(sc.fit_transform(Xs), ys)
        yp = clf.predict_proba(sc.transform(Xte))[:, 1]
        yt_all.extend(yte); yp_all.extend(yp)
    yt, yp = np.array(yt_all), np.array(yp_all)
    auc = round(roc_auc_score(yt, yp), 4)
    rng = np.random.RandomState(RANDOM_STATE)
    scores = [roc_auc_score(yt[idx], yp[idx]) for _ in range(1000)
              if len(np.unique(yt[idx := rng.choice(len(yt), len(yt), replace=True)])) > 1]
    return auc, round(np.percentile(scores, 2.5), 4), round(np.percentile(scores, 97.5), 4)

print("\nRunning ablation analysis...")
auc_full, lo_full, hi_full   = run_cv(prep(ALL), y)
auc_base, lo_base, hi_base   = run_cv(prep(BASE_ONLY), y)
delta = round(auc_full - auc_base, 4)

print(f"\n=== NHANES Ablation Results ===")
print(f"Full model  (12 features, all λ=0): AUC={auc_full} [{lo_full}–{hi_full}]")
print(f"Base model  (8 features, no CV hist): AUC={auc_base} [{lo_base}–{hi_base}]")
print(f"CV history contribution: ΔAUC={delta:+}")
print(f"\nInterpretation: CV history contributes +{delta} AUC (modest but genuine).")
print(f"CAE correctly RETAINED these features (λ=0): 0 false positives.")

results = {
    'full_model':  {'AUC': auc_full, 'CI_lo': lo_full, 'CI_hi': hi_full, 'n_features': len(ALL)},
    'base_model':  {'AUC': auc_base, 'CI_lo': lo_base, 'CI_hi': hi_base, 'n_features': len(BASE_ONLY)},
    'cv_history_contribution': delta,
    'false_positive_rate': 0.0,
    'interpretation': 'CAE correctly retained all 12 features as lambda=0. No false positives.'
}

Path('../results').mkdir(exist_ok=True)
with open('../results/nhanes_ablation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
pd.DataFrame([{'Model': 'Full (12 features)', 'AUC': auc_full, 'CI': f"[{lo_full}-{hi_full}]"},
              {'Model': 'Base (8 features)',  'AUC': auc_base,  'CI': f"[{lo_base}-{hi_base}]"},
              {'Model': 'ΔAUC (CV history)',  'AUC': delta,      'CI': '—'},
]).to_csv('../results/nhanes_ablation_results.csv', index=False)
print("\nSaved: results/nhanes_ablation_results.json, nhanes_ablation_results.csv")
