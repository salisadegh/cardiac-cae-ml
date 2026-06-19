"""
01_uci_heart_failure_leakage.py
================================
CAE validation on UCI Heart Failure dataset (N=299).
Reproduces Table 2 (UCI rows), Figure 2, Figure 3.

Dataset: Chicco & Jurman (2020), BMC Medical Informatics
Download: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records

Usage:
    python 01_uci_heart_failure_leakage.py

Output:
    results/uci_classifier_results.csv
    results/uci_inflation_results.csv
"""

import numpy as np
import pandas as pd
import json
import warnings
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier,
                               HistGradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────
# CAE Classification
# ─────────────────────────────────────────────────────────────────
# T₀ = baseline clinical assessment at enrolment
# "time" = follow-up duration → unavailable at T₀ → Q₁ logical equivalence → λ=3
# All other 11 features → λ=0 (available at T₀)

CAE_CLASSIFICATION = {
    'age':                       {'lambda': 0, 'action': 'retain',     'reason': 'demographic, λ=0'},
    'anaemia':                   {'lambda': 0, 'action': 'retain',     'reason': 'clinical status at T₀, λ=0'},
    'creatinine_phosphokinase':  {'lambda': 0, 'action': 'retain',     'reason': 'lab value at T₀, λ=0'},
    'diabetes':                  {'lambda': 0, 'action': 'retain',     'reason': 'comorbidity, λ=0'},
    'ejection_fraction':         {'lambda': 0, 'action': 'retain',     'reason': 'echocardiographic at T₀, λ=0'},
    'high_blood_pressure':       {'lambda': 0, 'action': 'retain',     'reason': 'comorbidity, λ=0'},
    'platelets':                 {'lambda': 0, 'action': 'retain',     'reason': 'lab value at T₀, λ=0'},
    'serum_creatinine':          {'lambda': 0, 'action': 'retain',     'reason': 'lab value at T₀, λ=0'},
    'serum_sodium':              {'lambda': 0, 'action': 'retain',     'reason': 'lab value at T₀, λ=0'},
    'sex':                       {'lambda': 0, 'action': 'retain',     'reason': 'demographic, λ=0'},
    'smoking':                   {'lambda': 0, 'action': 'retain',     'reason': 'history, λ=0'},
    'time':                      {'lambda': 3, 'action': 'remove',     'reason': 'post-baseline follow-up duration; unavailable at T₀; Q₁ logical equivalence → λ=3 REMOVE'},
}

print("=== CAE Feature Classification: UCI Heart Failure ===")
print(f"{'Feature':<30} {'λ':<4} {'Action':<20} Reason")
print("-" * 80)
for feat, info in CAE_CLASSIFICATION.items():
    print(f"{feat:<30} {info['lambda']:<4} {info['action']:<20} {info['reason']}")

LEAKY    = [f for f, v in CAE_CLASSIFICATION.items() if v['lambda'] == 3]
RETAINED = [f for f, v in CAE_CLASSIFICATION.items() if v['lambda'] == 0]
print(f"\nRemoved (λ=3): {LEAKY}")
print(f"Retained (λ=0): {len(RETAINED)} features\n")

# ─────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────
DATA_PATH = Path('../data/heart_failure_clinical_records.csv')
if not DATA_PATH.exists():
    # Try direct download
    import urllib.request
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv'
    try:
        urllib.request.urlretrieve(url, DATA_PATH)
        print(f"Downloaded dataset to {DATA_PATH}")
    except:
        print(f"Please download the dataset from https://archive.ics.uci.edu/dataset/519/")
        print(f"and save as: {DATA_PATH}")
        exit(1)

df  = pd.read_csv(DATA_PATH)
y   = df['DEATH_EVENT'].values.astype(int)
ALL = list(CAE_CLASSIFICATION.keys())
X_naive = df[ALL].values.astype(float)
X_cae   = df[RETAINED].values.astype(float)
print(f"Dataset: N={len(df)}, Deaths={y.sum()} ({y.mean()*100:.1f}%)")

# ─────────────────────────────────────────────────────────────────
# SMOTE (corrected: applied inside inner fold only)
# ─────────────────────────────────────────────────────────────────
def smote_oversample(X, y, k=3, random_state=42):
    """SMOTE — call ONLY inside inner training fold."""
    rng = np.random.RandomState(random_state)
    X, y = np.array(X, dtype=float), np.array(y)
    classes, counts = np.unique(y, return_counts=True)
    minority = classes[np.argmin(counts)]
    X_min = X[y == minority]
    n_syn = counts.max() - counts.min()
    synthetic = []
    for _ in range(n_syn):
        idx = rng.randint(0, len(X_min)); sample = X_min[idx]
        dists = np.sqrt(((X_min - sample)**2).sum(axis=1)); dists[idx] = np.inf
        k_eff = min(k, len(X_min) - 1)
        nb = X_min[np.argsort(dists)[:k_eff]][rng.randint(0, k_eff)]
        synthetic.append(sample + rng.rand() * (nb - sample))
    return np.vstack([X, np.array(synthetic)]), np.concatenate([y, np.full(n_syn, minority)])

# ─────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────
def get_models():
    return {
        'SVM (Linear)':       (lambda p: SVC(kernel='linear', C=p['C'], probability=True, class_weight='balanced', random_state=RANDOM_STATE), [{'C': c} for c in [0.01, 0.1, 1, 10]]),
        'SVM (RBF)':          (lambda p: SVC(kernel='rbf', C=p['C'], gamma=p['g'], probability=True, class_weight='balanced', random_state=RANDOM_STATE), [{'C': c, 'g': g} for c in [0.1, 1, 10] for g in ['scale', 'auto']]),
        'Logistic Regression':(lambda p: LogisticRegression(C=p['C'], max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE), [{'C': c} for c in [0.01, 0.1, 1, 10]]),
        'Random Forest':      (lambda p: RandomForestClassifier(n_estimators=200, max_depth=p['d'], class_weight='balanced', random_state=RANDOM_STATE), [{'d': d} for d in [3, 5, None]]),
        'Gradient Boosting':  (lambda p: GradientBoostingClassifier(n_estimators=200, learning_rate=p['lr'], max_depth=3, random_state=RANDOM_STATE), [{'lr': lr} for lr in [0.01, 0.05, 0.1, 0.2]]),
        'Hist-GBM':           (lambda p: HistGradientBoostingClassifier(max_iter=200, learning_rate=p['lr'], max_depth=4, random_state=RANDOM_STATE), [{'lr': lr} for lr in [0.01, 0.05, 0.1, 0.2]]),
        'AdaBoost':           (lambda p: AdaBoostClassifier(n_estimators=p['n'], learning_rate=0.1, random_state=RANDOM_STATE), [{'n': n} for n in [50, 100, 200]]),
        'Extra Trees':        (lambda p: ExtraTreesClassifier(n_estimators=200, max_depth=p['d'], class_weight='balanced', random_state=RANDOM_STATE), [{'d': d} for d in [3, 5, None]]),
        'Decision Tree':      (lambda p: DecisionTreeClassifier(max_depth=p['d'], class_weight='balanced', random_state=RANDOM_STATE), [{'d': d} for d in [3, 5, 7, None]]),
        'KNN':                (lambda p: KNeighborsClassifier(n_neighbors=p['k']), [{'k': k} for k in [3, 5, 7, 11]]),
        'GaussianNB':         (lambda p: GaussianNB(), [{}]),
        'LDA':                (lambda p: LinearDiscriminantAnalysis(), [{}]),
    }

# ─────────────────────────────────────────────────────────────────
# Nested CV
# ─────────────────────────────────────────────────────────────────
def nested_cv(factory, grid, X, y, n_outer=5, n_inner=3):
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=RANDOM_STATE)
    yt_all, yp_all = [], []
    for fold, (tr, te) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=fold)
        best_auc, best_p = -1, grid[0]
        for params in grid:
            scores = []
            for itr, ival in inner_cv.split(X_tr, y_tr):
                k = min(3, max(1, int((y_tr[itr] == 0).sum()) - 1))
                if y_tr[itr].sum() >= 3 and (y_tr[itr] == 0).sum() >= 3:
                    Xi, yi = smote_oversample(X_tr[itr], y_tr[itr], k=k, random_state=fold)
                else:
                    Xi, yi = X_tr[itr], y_tr[itr]
                sc = StandardScaler(); clf = factory(params)
                clf.fit(sc.fit_transform(Xi), yi)
                if len(np.unique(y_tr[ival])) > 1:
                    yp = clf.predict_proba(sc.transform(X_tr[ival]))[:, 1] if hasattr(clf, 'predict_proba') else clf.decision_function(X_tr[ival])
                    scores.append(roc_auc_score(y_tr[ival], yp))
            if scores and np.mean(scores) > best_auc:
                best_auc = np.mean(scores); best_p = params
        k = min(3, max(1, int((y_tr == 0).sum()) - 1))
        if y_tr.sum() >= 3 and (y_tr == 0).sum() >= 3:
            Xs, ys = smote_oversample(X_tr, y_tr, k=k, random_state=fold)
        else:
            Xs, ys = X_tr, y_tr
        sc = StandardScaler(); clf = factory(best_p)
        clf.fit(sc.fit_transform(Xs), ys)
        yp = clf.predict_proba(sc.transform(X_te))[:, 1] if hasattr(clf, 'predict_proba') else clf.decision_function(X_te)
        yt_all.extend(y_te); yp_all.extend(yp)
    return np.array(yt_all), np.array(yp_all)

def bootstrap_ci(yt, yp, n=2000):
    rng = np.random.RandomState(RANDOM_STATE)
    scores = [roc_auc_score(yt[idx], yp[idx])
              for _ in range(n)
              if len(np.unique(yt[idx := rng.choice(len(yt), len(yt), replace=True)])) > 1]
    return round(np.percentile(scores, 2.5), 4), round(np.percentile(scores, 97.5), 4)

# ─────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────
models = get_models()
results = {}
print("Running nested 5×3-fold CV for all 12 classifiers...")
print(f"{'Pipeline':<20} {'Model':<25} AUC  [95% CI]")
print("-" * 65)

for pipeline, X in [('Naive', X_naive), ('CAE-Cleaned', X_cae)]:
    results[pipeline] = {}
    for name, (factory, grid) in models.items():
        yt, yp = nested_cv(factory, grid, X, y)
        auc = round(roc_auc_score(yt, yp), 4)
        lo, hi = bootstrap_ci(yt, yp)
        brier = round(brier_score_loss(yt, yp), 4)
        results[pipeline][name] = {'AUC': auc, 'CI_lo': lo, 'CI_hi': hi, 'Brier': brier}
        print(f"{pipeline:<20} {name:<25} {auc:.4f} [{lo}-{hi}]")

# ─────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────
print("\n=== AUC Inflation Summary ===")
rows = []
for name in models:
    n = results['Naive'][name]
    c = results['CAE-Cleaned'][name]
    infl = round(n['AUC'] - c['AUC'], 4)
    rows.append({'Model': name, 'Naive_AUC': n['AUC'], 'Naive_CI': f"[{n['CI_lo']}-{n['CI_hi']}]",
                 'CAE_AUC': c['AUC'], 'CAE_CI': f"[{c['CI_lo']}-{c['CI_hi']}]",
                 'Inflation': infl, 'CAE_Brier': c['Brier']})
    print(f"  {name:<25} Naive={n['AUC']:.4f}  CAE={c['AUC']:.4f}  Δ={infl:+.4f}")

df_out = pd.DataFrame(rows).sort_values('CAE_AUC', ascending=False)
inflations = [r['Inflation'] for r in rows]
print(f"\n  Mean inflation: {np.mean(inflations):+.4f}")
print(f"  Range: {min(inflations):+.4f} to {max(inflations):+.4f}")

Path('../results').mkdir(exist_ok=True)
df_out.to_csv('../results/uci_classifier_results.csv', index=False)
with open('../results/uci_all_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nSaved: results/uci_classifier_results.csv, uci_all_results.json")
