"""
sensitivity_analysis.py
------------------------
Four-tier progressive sensitivity analysis and leave-one-out marginal
decomposition for the contested λ=1 variables (ECMO, CRRT, Furosemide).
Also includes the Furosemide ordinal-encoding comparison.

Author: Dr. Ali Sadegh-Zadeh
        Staffordshire University

Usage:
    python sensitivity_analysis.py --data data/synthetic_cardiac_cae_public.csv

Output:
    results/progressive_sensitivity.csv
    results/loo_marginal.csv
    results/furosemide_encoding.csv

Notes
-----
Furosemide in the original cardiac surgery dataset takes only three ordinal
values: {0 = none, 1 = standard, 100 = intensive}. It is NOT a continuous
dose measurement in mg. All interpretations reflect this ordinal structure.

Leave-one-out differences should be interpreted cautiously given the small
sample size (N=113, minority class n=16). These are within-pipeline
descriptive findings, not definitive claims about individual variable
contributions.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from analysis_main import (bootstrap_auc_ci, load_and_prepare,
                            smote_oversample)

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ─────────────────────────────────────────────────────────────────
# Minimal SVM nested CV (SVM-Linear only, for speed)
# ─────────────────────────────────────────────────────────────────

def svm_nested_cv(X, y, C_grid=None, n_outer=5, n_inner=3):
    if C_grid is None:
        C_grid = [0.01, 0.1, 1.0, 10.0]
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True,
                                random_state=RANDOM_STATE)
    y_true_all, y_prob_all = [], []
    for fold_idx, (tr, te) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True,
                                    random_state=fold_idx)
        best_auc, best_C = -1, 1.0
        for C_ in C_grid:
            inner_aucs = []
            for itr, ival in inner_cv.split(X_tr, y_tr):
                k = min(3, max(1, int((y_tr[itr] == 0).sum()) - 1))
                if y_tr[itr].sum() >= 3 and (y_tr[itr] == 0).sum() >= 3:
                    Xi_sm, yi_sm = smote_oversample(X_tr[itr], y_tr[itr],
                                                     k=k, random_state=fold_idx)
                else:
                    Xi_sm, yi_sm = X_tr[itr], y_tr[itr]
                sc = StandardScaler()
                clf = SVC(kernel='linear', C=C_, probability=True,
                          class_weight='balanced', random_state=RANDOM_STATE)
                clf.fit(sc.fit_transform(Xi_sm), yi_sm)
                if len(np.unique(y_tr[ival])) > 1:
                    inner_aucs.append(
                        roc_auc_score(y_tr[ival],
                                      clf.predict_proba(sc.transform(X_tr[ival]))[:, 1])
                    )
            if inner_aucs and np.mean(inner_aucs) > best_auc:
                best_auc = np.mean(inner_aucs)
                best_C = C_
        k = min(3, max(1, int((y_tr == 0).sum()) - 1))
        if y_tr.sum() >= 3 and (y_tr == 0).sum() >= 3:
            X_sm, y_sm = smote_oversample(X_tr, y_tr, k=k, random_state=fold_idx)
        else:
            X_sm, y_sm = X_tr, y_tr
        sc = StandardScaler()
        clf = SVC(kernel='linear', C=best_C, probability=True,
                  class_weight='balanced', random_state=RANDOM_STATE)
        clf.fit(sc.fit_transform(X_sm), y_sm)
        y_prob = clf.predict_proba(sc.transform(X_te))[:, 1]
        y_true_all.extend(y_te)
        y_prob_all.extend(y_prob)
    return np.array(y_true_all), np.array(y_prob_all)


def run_and_report(X, y, label, n_feats):
    yt, yp = svm_nested_cv(X, y)
    auc = roc_auc_score(yt, yp)
    lo, hi = bootstrap_auc_ci(yt, yp, n_bootstrap=1000)
    print(f"  {label:<50} N_feats={n_feats:2d}  AUC={auc:.4f} [{lo:.4f}–{hi:.4f}]")
    return {'label': label, 'N_features': n_feats,
            'AUC': round(auc, 4), 'CI_lo': lo, 'CI_hi': hi}


# ─────────────────────────────────────────────────────────────────
# Progressive sensitivity analysis
# ─────────────────────────────────────────────────────────────────

def progressive_analysis(df_full, y):
    """
    Four-tier progressive feature inclusion (SVM Linear, nested CV).

    Tier-0: CAE=0 only (35 pre-operative features)
    Tier-1: CAE=0 + λ=1 excluding ECMO, CRRT, Furosemide (51 features)
    Tier-2: CAE=0 + all λ=1 except Furosemide (53 features)
    Tier-3: All 54 features (CAE=0 + all CAE=1)
    """
    from cae_algorithm import build_cardiac_surgery_cae
    cae = build_cardiac_surgery_cae()
    all_features = [c for c in df_full.columns if c != 'One_Year_Survival']
    cae.classify_all(all_features)
    cae0 = [f for f in all_features
            if cae._results.get(f) and cae._results[f].lambda_score == 0]
    cae1 = [f for f in all_features
            if cae._results.get(f) and cae._results[f].lambda_score == 1]
    contested = ['ECMO', 'CRRT', 'Furosemide']

    def prep(feats):
        X = df_full[feats].copy()
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = X[col].fillna(X[col].mode()[0])
            else:
                X[col] = X[col].fillna(X[col].median())
        return X.values.astype(float)

    tiers = [
        ('Tier-0: CAE=0 only (pre-operative)',
         cae0),
        ('Tier-1: CAE=0 + λ=1 excl. ECMO/CRRT/Furosemide',
         cae0 + [f for f in cae1 if f not in contested]),
        ('Tier-2: CAE=0 + λ=1 excl. Furosemide only',
         cae0 + [f for f in cae1 if f != 'Furosemide']),
        ('Tier-3: All 54 features (CAE=0 + all CAE=1)',
         cae0 + cae1),
    ]

    rows = []
    print("=== Progressive Sensitivity Analysis ===")
    for label, feats in tiers:
        row = run_and_report(prep(feats), y, label, len(feats))
        rows.append(row)

    # Delta vs Tier-0
    t0_auc = rows[0]['AUC']
    for row in rows[1:]:
        row['delta_vs_T0'] = round(row['AUC'] - t0_auc, 4)
    rows[0]['delta_vs_T0'] = 0.0
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Leave-one-out marginal analysis
# ─────────────────────────────────────────────────────────────────

def loo_analysis(df_full, y, base_features):
    """
    Remove each contested variable individually from the full feature set.

    Interpretation note: these are within-pipeline descriptive statistics.
    Given N=113 and 16 deaths, differences may not be stable across
    pipelines or cohorts.
    """
    def prep(feats):
        X = df_full[feats].copy()
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = X[col].fillna(X[col].mode()[0])
            else:
                X[col] = X[col].fillna(X[col].median())
        return X.values.astype(float)

    print("\n=== Leave-One-Out Marginal Analysis (within-pipeline) ===")
    print("  (Differences should be interpreted cautiously given N=113)")

    # Baseline: all features
    yt, yp = svm_nested_cv(prep(base_features), y)
    base_auc = roc_auc_score(yt, yp)
    base_lo, base_hi = bootstrap_auc_ci(yt, yp, n_bootstrap=1000)
    print(f"  {'Baseline (all features)':<50} AUC={base_auc:.4f} [{base_lo:.4f}–{base_hi:.4f}]")
    rows = [{'variable_removed': 'none (baseline)', 'N_features': len(base_features),
             'AUC': round(base_auc, 4), 'CI_lo': base_lo, 'CI_hi': base_hi,
             'delta_vs_baseline': 0.0, 'caution': 'N/A'}]

    for var in ['ECMO', 'CRRT', 'Furosemide']:
        if var not in base_features:
            continue
        feats_minus = [f for f in base_features if f != var]
        yt, yp = svm_nested_cv(prep(feats_minus), y)
        auc = roc_auc_score(yt, yp)
        lo, hi = bootstrap_auc_ci(yt, yp, n_bootstrap=1000)
        delta = round(base_auc - auc, 4)
        print(f"  {'Remove ' + var:<50} AUC={auc:.4f} [{lo:.4f}–{hi:.4f}]  Δ={delta:+.4f}")
        rows.append({'variable_removed': var, 'N_features': len(feats_minus),
                     'AUC': round(auc, 4), 'CI_lo': lo, 'CI_hi': hi,
                     'delta_vs_baseline': delta,
                     'caution': 'interpret cautiously; N=113, pipeline-specific'})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Furosemide ordinal-encoding comparison
# ─────────────────────────────────────────────────────────────────

def furosemide_encoding_analysis(df_full, y, base_features):
    """
    Compare four Furosemide encodings within the same nested CV:
      A: ordinal as-coded {0, 1, 100}
      B: binary flag (any vs. none)
      C: quintile bins
      D: absent (Tier-2)

    Note: Furosemide in this dataset has only 3 values {0, 1, 100}.
    It is an ordinal variable, not a continuous dose measurement.
    """
    def prep_custom(df_mod, feats):
        X = df_mod[feats].copy()
        for col in X.columns:
            if X[col].dtype == object:
                X[col] = X[col].fillna(X[col].mode()[0])
            else:
                X[col] = X[col].fillna(X[col].median())
        return X.values.astype(float)

    print("\n=== Furosemide Ordinal-Encoding Analysis ===")
    print("  Note: Furosemide takes only 3 values {0, 1, 100} in original data.")
    print("  Model A = ordinal as-coded, B = binary flag, C = quintile, D = absent")

    rows = []
    non_furo = [f for f in base_features if f != 'Furosemide']

    # A: ordinal as-coded
    yt, yp = svm_nested_cv(prep_custom(df_full, base_features), y)
    auc_a = roc_auc_score(yt, yp)
    lo_a, hi_a = bootstrap_auc_ci(yt, yp, n_bootstrap=1000)
    print(f"  Model A (ordinal {'{0,1,100}'}):   AUC={auc_a:.4f} [{lo_a:.4f}–{hi_a:.4f}]")
    rows.append({'model': 'A_ordinal_as_coded', 'AUC': round(auc_a, 4),
                 'CI_lo': lo_a, 'CI_hi': hi_a, 'delta_vs_A': 0.0})

    # B: binary
    df_b = df_full.copy()
    df_b['Furosemide_binary'] = (df_b['Furosemide'].fillna(0) > 0).astype(int)
    feats_b = non_furo + ['Furosemide_binary']
    yt, yp = svm_nested_cv(prep_custom(df_b, feats_b), y)
    auc_b = roc_auc_score(yt, yp)
    lo_b, hi_b = bootstrap_auc_ci(yt, yp, n_bootstrap=1000)
    print(f"  Model B (binary flag):       AUC={auc_b:.4f} [{lo_b:.4f}–{hi_b:.4f}]  Δ={auc_a-auc_b:+.4f}")
    rows.append({'model': 'B_binary_flag', 'AUC': round(auc_b, 4),
                 'CI_lo': lo_b, 'CI_hi': hi_b, 'delta_vs_A': round(auc_a - auc_b, 4)})

    # C: quintile
    df_c = df_full.copy()
    df_c['Furosemide_quintile'] = pd.qcut(
        df_c['Furosemide'].fillna(0), q=5, labels=False, duplicates='drop')
    feats_c = non_furo + ['Furosemide_quintile']
    yt, yp = svm_nested_cv(prep_custom(df_c, feats_c), y)
    auc_c = roc_auc_score(yt, yp)
    lo_c, hi_c = bootstrap_auc_ci(yt, yp, n_bootstrap=1000)
    print(f"  Model C (quintile bins):     AUC={auc_c:.4f} [{lo_c:.4f}–{hi_c:.4f}]  Δ={auc_a-auc_c:+.4f}")
    rows.append({'model': 'C_quintile', 'AUC': round(auc_c, 4),
                 'CI_lo': lo_c, 'CI_hi': hi_c, 'delta_vs_A': round(auc_a - auc_c, 4)})

    # D: absent (Tier-2)
    yt, yp = svm_nested_cv(prep_custom(df_full, non_furo), y)
    auc_d = roc_auc_score(yt, yp)
    lo_d, hi_d = bootstrap_auc_ci(yt, yp, n_bootstrap=1000)
    print(f"  Model D (Furosemide absent): AUC={auc_d:.4f} [{lo_d:.4f}–{hi_d:.4f}]  Δ={auc_a-auc_d:+.4f}")
    rows.append({'model': 'D_absent_Tier2', 'AUC': round(auc_d, 4),
                 'CI_lo': lo_d, 'CI_hi': hi_d, 'delta_vs_A': round(auc_a - auc_d, 4)})

    print(f"\n  Key gap (ordinal vs binary): {auc_a - auc_b:+.4f}")
    print(f"  Key gap (binary vs absent):  {auc_b - auc_d:+.4f}")
    print(f"  Conservative estimate (Tier-2 / Model D): AUC={auc_d:.4f} [{lo_d:.4f}–{hi_d:.4f}]")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main(data_path: str, output_dir: str = 'results'):
    Path(output_dir).mkdir(exist_ok=True)

    df = pd.read_csv(data_path).dropna(subset=['One_Year_Survival'])
    le = LabelEncoder()
    if 'Tx_Etiology' in df.columns:
        df['Tx_Etiology'] = le.fit_transform(df['Tx_Etiology'].astype(str))
    y = df['One_Year_Survival'].values.astype(int)

    # Impute missing values
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna(df[col].mode()[0])
        elif col != 'One_Year_Survival':
            df[col] = df[col].fillna(df[col].median())

    from cae_algorithm import build_cardiac_surgery_cae
    cae = build_cardiac_surgery_cae()
    all_features = [c for c in df.columns if c != 'One_Year_Survival']
    cae.classify_all(all_features)
    base_features = cae.get_retained(include_annotated=True)
    base_features = [f for f in base_features if f in df.columns]
    print(f"Dataset: N={len(df)}, retained features={len(base_features)}\n")

    prog_df = progressive_analysis(df, y)
    prog_df.to_csv(f'{output_dir}/progressive_sensitivity.csv', index=False)

    loo_df = loo_analysis(df, y, base_features)
    loo_df.to_csv(f'{output_dir}/loo_marginal.csv', index=False)

    furo_df = furosemide_encoding_analysis(df, y, base_features)
    furo_df.to_csv(f'{output_dir}/furosemide_encoding.csv', index=False)

    print(f"\nAll sensitivity results saved to {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/synthetic_cardiac_cae_public.csv')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    main(args.data, args.output)
