"""
simulation_study.py
--------------------
Domain-independent simulation study to validate the CAE leakage-detection
procedure across a range of dataset configurations and leakage severities.

Author: Dr. Ali Sadegh-Zadeh
        Staffordshire University

Usage:
    python simulation_study.py

Output:
    results/simulation_results.csv
    results/simulation_summary.txt

Method
------
For each leakage level (0%, 16.7%, 28.6%), 120 synthetic binary-classification
datasets are generated. Each dataset is evaluated with two pipelines:
  (a) Naive pipeline: no CAE filtering (leaky features included)
  (b) CAE-cleaned pipeline: leaky features removed before training

SVM (Linear) with 5-fold stratified CV is used as the base estimator.
AUC-ROC is the primary outcome measure.

The null hypothesis is that naive and CAE-cleaned pipelines yield identical
AUC (i.e., leakage causes no inflation). A one-sided Wilcoxon signed-rank
test is used; the alternative hypothesis is that naive AUC > CAE-cleaned AUC.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ─────────────────────────────────────────────────────────────────
# Synthetic dataset generation
# ─────────────────────────────────────────────────────────────────

def make_dataset_with_leakage(n_samples=113, n_features=54,
                               n_leaky=0, seed=0):
    """
    Generate a synthetic binary classification dataset.

    Leaky features are added as outcome-correlated variables that would
    not be available at the prediction timepoint T0 in a real deployment.
    They are NOT genuine predictors — they encode post-T0 information.

    Parameters
    ----------
    n_samples : int
    n_features : int — number of legitimate features
    n_leaky : int — number of leaky (outcome-correlated) features to inject
    seed : int

    Returns
    -------
    X : ndarray (n_samples, n_features + n_leaky)
    y : ndarray (n_samples,)
    leaky_indices : list of int — column indices of leaky features
    """
    rng = np.random.RandomState(seed)
    X_clean, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=4,
        n_classes=2,
        weights=[0.86, 0.14],
        random_state=seed,
        class_sep=0.8,
    )
    leaky_indices = []
    if n_leaky > 0:
        noise_scale = 0.3
        leaky_cols = []
        for i in range(n_leaky):
            leaky = y.astype(float) + rng.normal(0, noise_scale, size=len(y))
            leaky_cols.append(leaky.reshape(-1, 1))
        X_leaky = np.hstack([X_clean] + leaky_cols)
        leaky_indices = list(range(n_features, n_features + n_leaky))
    else:
        X_leaky = X_clean
    return X_leaky, y, leaky_indices


# ─────────────────────────────────────────────────────────────────
# Cross-validation
# ─────────────────────────────────────────────────────────────────

def cv_auc(X, y, n_folds=5, seed=0):
    """5-fold stratified CV, SVM Linear, returns mean AUC."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in cv.split(X, y):
        sc = StandardScaler()
        clf = SVC(kernel='linear', C=1.0, probability=True,
                  class_weight='balanced', random_state=RANDOM_STATE)
        clf.fit(sc.fit_transform(X[tr]), y[tr])
        if len(np.unique(y[te])) > 1:
            aucs.append(roc_auc_score(y[te],
                                       clf.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(aucs)) if aucs else np.nan


# ─────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────

def run_simulation(n_datasets=120, n_samples=113, n_features=54,
                   output_dir='results'):
    """
    Run the full simulation across three leakage levels.

    Leakage levels:
      0%   — no leaky features (control condition)
      16.7% — 9 leaky features / 54 total
      28.6% — 15 leaky features / 54 total
    """
    Path(output_dir).mkdir(exist_ok=True)
    configs = [
        {'label': '0% leakage (control)', 'n_leaky': 0},
        {'label': '16.7% leakage',         'n_leaky': 9},
        {'label': '28.6% leakage',         'n_leaky': 15},
    ]
    all_rows = []
    summary_lines = [
        "CAE Simulation Study — Domain-Independent Validation",
        f"Author: Dr. Ali Sadegh-Zadeh, Staffordshire University",
        f"N datasets per condition: {n_datasets}",
        f"N samples per dataset: {n_samples}",
        f"Classifier: SVM (Linear), 5-fold CV",
        "=" * 60,
    ]

    for config in configs:
        naive_aucs, clean_aucs = [], []
        for seed in range(n_datasets):
            X, y, leaky_idx = make_dataset_with_leakage(
                n_samples=n_samples,
                n_features=n_features,
                n_leaky=config['n_leaky'],
                seed=seed,
            )
            # Naive: all features including leaky
            naive = cv_auc(X, y, seed=seed)
            # CAE-cleaned: remove leaky features
            clean_cols = [i for i in range(X.shape[1]) if i not in leaky_idx]
            clean = cv_auc(X[:, clean_cols], y, seed=seed)
            naive_aucs.append(naive)
            clean_aucs.append(clean)
            all_rows.append({
                'config': config['label'],
                'seed': seed,
                'naive_auc': round(naive, 4),
                'clean_auc': round(clean, 4),
                'inflation': round(naive - clean, 4),
            })

        naive_aucs = np.array(naive_aucs)
        clean_aucs = np.array(clean_aucs)
        inflation = naive_aucs - clean_aucs
        mean_inf = inflation.mean()
        sd_inf = inflation.std()

        if config['n_leaky'] > 0 and (inflation != 0).any():
            stat, p_val = wilcoxon(naive_aucs, clean_aucs, alternative='greater')
        else:
            stat, p_val = np.nan, np.nan

        line = (
            f"\n{config['label']}\n"
            f"  Mean AUC  — Naive:   {naive_aucs.mean():.4f} ± {naive_aucs.std():.4f}\n"
            f"  Mean AUC  — CAE:     {clean_aucs.mean():.4f} ± {clean_aucs.std():.4f}\n"
            f"  Inflation (Naive−CAE): {mean_inf:+.4f} ± {sd_inf:.4f}\n"
            f"  Wilcoxon p (one-sided, Naive>CAE): {p_val if not np.isnan(p_val) else 'N/A (0 leakage)'}\n"
        )
        summary_lines.append(line)
        print(line)

    # Save
    df = pd.DataFrame(all_rows)
    df.to_csv(f'{output_dir}/simulation_results.csv', index=False)
    with open(f'{output_dir}/simulation_summary.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"\nSimulation results saved to {output_dir}/")


if __name__ == '__main__':
    run_simulation()
