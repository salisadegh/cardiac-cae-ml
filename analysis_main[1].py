"""
analysis_main.py
----------------
Full 12-model nested cross-validation pipeline for one-year survival
prediction after cardiac surgery, incorporating the CAE leakage-detection
procedure.

Author: Dr. Ali Sadegh-Zadeh
        Staffordshire University

Usage:
    python analysis_main.py --data data/synthetic_cardiac_cae_public.csv
    python analysis_main.py --data /path/to/real_data.csv

Output:
    results/model_performance.csv
    results/calibration_metrics.csv
    results/roc_data.json
    results/pr_data.json

Reference:
    Sadegh-Zadeh, A. (2025). Causal Adjacency Examination (CAE): A Formalised
    Leakage-Detection Procedure for Clinical Machine Learning Pipelines.
    Artificial Intelligence Review [under review].
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                               GradientBoostingClassifier,
                               HistGradientBoostingClassifier,
                               RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                              matthews_corrcoef, precision_recall_curve,
                              roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from cae_algorithm import build_cardiac_surgery_cae

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ─────────────────────────────────────────────────────────────────
# SMOTE — minority over-sampling (inside inner folds only)
# ─────────────────────────────────────────────────────────────────

def smote_oversample(X, y, k=3, random_state=42):
    """
    Synthetic Minority Over-sampling Technique (Chawla et al., 2002).

    Applied ONLY inside each inner training fold of the nested CV.
    Validation and test folds are never augmented — this prevents
    the contamination described in the manuscript Section 3.5.

    Parameters
    ----------
    X : ndarray, shape (n, d)
    y : ndarray, shape (n,)
    k : int — number of nearest neighbours for synthesis
    random_state : int

    Returns
    -------
    X_resampled, y_resampled : ndarrays with minority class augmented
    """
    rng = np.random.RandomState(random_state)
    X, y = np.array(X, dtype=float), np.array(y)
    classes, counts = np.unique(y, return_counts=True)
    minority_class = classes[np.argmin(counts)]
    X_min = X[y == minority_class]
    n_synthetic = counts.max() - counts.min()
    synthetic = []
    for _ in range(n_synthetic):
        idx = rng.randint(0, len(X_min))
        sample = X_min[idx]
        dists = np.sqrt(((X_min - sample) ** 2).sum(axis=1))
        dists[idx] = np.inf
        neighbours = X_min[np.argsort(dists)[:k]]
        neighbour = neighbours[rng.randint(0, len(neighbours))]
        synthetic.append(sample + rng.rand() * (neighbour - sample))
    X_out = np.vstack([X, np.array(synthetic)])
    y_out = np.concatenate([y, np.full(n_synthetic, minority_class)])
    return X_out, y_out


# ─────────────────────────────────────────────────────────────────
# Model registry with hyperparameter grids
# ─────────────────────────────────────────────────────────────────

def get_model_registry():
    """
    Returns dict: model_name → (factory_fn, param_grid).

    Hyperparameter grids are kept deliberately small given N=113.
    Wider grids would increase computational cost without meaningful
    improvement given the outer-fold evaluation sample sizes.
    """
    return {
        'SVM (Linear)': (
            lambda p: SVC(kernel='linear', C=p['C'], probability=True,
                          class_weight='balanced', random_state=RANDOM_STATE),
            [{'C': c} for c in [0.01, 0.1, 1.0, 10.0]]
        ),
        'SVM (RBF)': (
            lambda p: SVC(kernel='rbf', C=p['C'], gamma=p['gamma'],
                          probability=True, class_weight='balanced',
                          random_state=RANDOM_STATE),
            [{'C': c, 'gamma': g}
             for c in [0.1, 1.0, 10.0]
             for g in ['scale', 'auto']]
        ),
        'Logistic Regression': (
            lambda p: LogisticRegression(C=p['C'], max_iter=1000,
                                         class_weight='balanced',
                                         random_state=RANDOM_STATE),
            [{'C': c} for c in [0.01, 0.1, 1.0, 10.0]]
        ),
        'Random Forest': (
            lambda p: RandomForestClassifier(n_estimators=200,
                                              max_depth=p['max_depth'],
                                              class_weight='balanced',
                                              random_state=RANDOM_STATE),
            [{'max_depth': d} for d in [3, 5, None]]
        ),
        'Gradient Boosting': (
            lambda p: GradientBoostingClassifier(n_estimators=200,
                                                  learning_rate=p['lr'],
                                                  max_depth=3,
                                                  random_state=RANDOM_STATE),
            [{'lr': lr} for lr in [0.01, 0.05, 0.1, 0.2]]
        ),
        'Hist-GBM': (
            lambda p: HistGradientBoostingClassifier(max_iter=200,
                                                      learning_rate=p['lr'],
                                                      max_depth=4,
                                                      random_state=RANDOM_STATE),
            [{'lr': lr} for lr in [0.01, 0.05, 0.1, 0.2]]
        ),
        'AdaBoost': (
            lambda p: AdaBoostClassifier(n_estimators=p['n'],
                                          learning_rate=0.1,
                                          random_state=RANDOM_STATE),
            [{'n': n} for n in [50, 100, 200]]
        ),
        'Extra Trees': (
            lambda p: ExtraTreesClassifier(n_estimators=200,
                                            max_depth=p['max_depth'],
                                            class_weight='balanced',
                                            random_state=RANDOM_STATE),
            [{'max_depth': d} for d in [3, 5, None]]
        ),
        'Decision Tree': (
            lambda p: DecisionTreeClassifier(max_depth=p['max_depth'],
                                              class_weight='balanced',
                                              random_state=RANDOM_STATE),
            [{'max_depth': d} for d in [3, 5, 7, None]]
        ),
        'KNN': (
            lambda p: KNeighborsClassifier(n_neighbors=p['k']),
            [{'k': k} for k in [3, 5, 7, 11]]
        ),
        'GaussianNB': (
            lambda p: GaussianNB(),
            [{}]
        ),
        'LDA': (
            lambda p: LinearDiscriminantAnalysis(),
            [{}]
        ),
    }


# ─────────────────────────────────────────────────────────────────
# Nested cross-validation
# ─────────────────────────────────────────────────────────────────

def nested_cv(factory, grid, X, y, n_outer=5, n_inner=3):
    """
    Nested stratified k-fold cross-validation.

    Outer loop: performance estimation.
    Inner loop: hyperparameter selection.
    SMOTE: applied inside each inner training fold only.

    Returns
    -------
    y_true, y_prob : concatenated test-fold labels and probability scores
    """
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True,
                                random_state=RANDOM_STATE)
    y_true_all, y_prob_all = [], []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True,
                                    random_state=fold_idx)
        best_score, best_params = -1, grid[0]

        for params in grid:
            inner_scores = []
            for i_train, i_val in inner_cv.split(X_train, y_train):
                Xi_raw, yi_raw = X_train[i_train], y_train[i_train]
                Xv_raw, yv_raw = X_train[i_val], y_train[i_val]
                k_smote = min(3, max(1, int((yi_raw == 0).sum()) - 1))
                if yi_raw.sum() >= 3 and (yi_raw == 0).sum() >= 3:
                    Xi_sm, yi_sm = smote_oversample(Xi_raw, yi_raw,
                                                     k=k_smote,
                                                     random_state=fold_idx)
                else:
                    Xi_sm, yi_sm = Xi_raw, yi_raw
                scaler = StandardScaler()
                Xi_sc = scaler.fit_transform(Xi_sm)
                Xv_sc = scaler.transform(Xv_raw)
                clf = factory(params)
                clf.fit(Xi_sc, yi_sm)
                if len(np.unique(yv_raw)) > 1:
                    if hasattr(clf, 'predict_proba'):
                        yp = clf.predict_proba(Xv_sc)[:, 1]
                    else:
                        d = clf.decision_function(Xv_sc)
                        yp = (d - d.min()) / (d.max() - d.min() + 1e-8)
                    inner_scores.append(roc_auc_score(yv_raw, yp))
            if inner_scores and np.mean(inner_scores) > best_score:
                best_score = np.mean(inner_scores)
                best_params = params

        # Retrain on full outer training set with best params
        k_smote = min(3, max(1, int((y_train == 0).sum()) - 1))
        if y_train.sum() >= 3 and (y_train == 0).sum() >= 3:
            X_tr_sm, y_tr_sm = smote_oversample(X_train, y_train,
                                                  k=k_smote,
                                                  random_state=fold_idx)
        else:
            X_tr_sm, y_tr_sm = X_train, y_train

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_sm)
        X_te_sc = scaler.transform(X_test)
        clf = factory(best_params)
        clf.fit(X_tr_sc, y_tr_sm)

        if hasattr(clf, 'predict_proba'):
            y_prob = clf.predict_proba(X_te_sc)[:, 1]
        else:
            d = clf.decision_function(X_te_sc)
            y_prob = (d - d.min()) / (d.max() - d.min() + 1e-8)

        y_true_all.extend(y_test)
        y_prob_all.extend(y_prob)

    return np.array(y_true_all), np.array(y_prob_all)


# ─────────────────────────────────────────────────────────────────
# Evaluation utilities
# ─────────────────────────────────────────────────────────────────

def bootstrap_auc_ci(y_true, y_prob, n_bootstrap=2000, alpha=0.05):
    """Stratified bootstrap 95% CI for AUC-ROC."""
    rng = np.random.RandomState(RANDOM_STATE)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    return round(lo, 4), round(hi, 4)


def youden_threshold(y_true, y_prob):
    """Optimal threshold via Youden's J = max(Sensitivity + Specificity - 1)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return thresholds[np.argmax(j)]


def hosmer_lemeshow(y_true, y_prob, g=10):
    """
    Hosmer-Lemeshow χ² goodness-of-fit test (g equal-width bins).
    Returns (chi2_stat, p_value).
    Note: reduced power with small N; treat as a preliminary diagnostic only.
    """
    bins = np.linspace(0, 1, g + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, g - 1)
    chi2_stat = 0.0
    for b in range(g):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        obs_pos = y_true[mask].sum()
        exp_pos = y_prob[mask].sum()
        obs_neg = mask.sum() - obs_pos
        exp_neg = mask.sum() - exp_pos
        if exp_pos > 0:
            chi2_stat += (obs_pos - exp_pos) ** 2 / exp_pos
        if exp_neg > 0:
            chi2_stat += (obs_neg - exp_neg) ** 2 / exp_neg
    p_val = 1 - scipy_stats.chi2.cdf(chi2_stat, df=g - 2)
    return round(chi2_stat, 4), round(p_val, 4)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Expected Calibration Error (ECE) with equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        frac_pos = y_true[mask].mean()
        mean_pred = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(frac_pos - mean_pred)
    return round(ece, 4)


def evaluate_model(y_true, y_prob):
    """Compute full evaluation suite for a model."""
    auc = round(roc_auc_score(y_true, y_prob), 4)
    ci_lo, ci_hi = bootstrap_auc_ci(y_true, y_prob)
    brier = round(brier_score_loss(y_true, y_prob), 4)
    ece = expected_calibration_error(y_true, y_prob)
    hl_chi2, hl_p = hosmer_lemeshow(y_true, y_prob)
    thresh = youden_threshold(y_true, y_prob)
    y_pred = (y_prob >= thresh).astype(int)
    TP = int(((y_pred == 0) & (y_true == 0)).sum())
    TN = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 0) & (y_true == 1)).sum())
    FN = int(((y_pred == 1) & (y_true == 0)).sum())
    sens = round(TN / (TN + FN) if (TN + FN) > 0 else 0.0, 4)
    spec = round(TP / (TP + FP) if (TP + FP) > 0 else 0.0, 4)
    mcc = round(matthews_corrcoef(y_true, y_pred), 4)
    return {
        'AUC': auc, 'AUC_CI_lo': ci_lo, 'AUC_CI_hi': ci_hi,
        'Brier': brier, 'ECE': ece, 'HL_chi2': hl_chi2, 'HL_p': hl_p,
        'Sensitivity': sens, 'Specificity': spec, 'MCC': mcc,
        'Youden_threshold': round(thresh, 4),
    }


# ─────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────

def load_and_prepare(data_path: str):
    """
    Load CSV, apply CAE filtering, return (X, y, feature_names).
    Leaky feature (In_Hospital_Mortality) is removed via CAE (λ=3).
    """
    df = pd.read_csv(data_path).dropna(subset=['One_Year_Survival'])
    le = LabelEncoder()
    if 'Tx_Etiology' in df.columns:
        df['Tx_Etiology'] = le.fit_transform(df['Tx_Etiology'].astype(str))

    # CAE classification
    cae = build_cardiac_surgery_cae()
    all_features = [c for c in df.columns if c != 'One_Year_Survival']
    cae.classify_all(all_features)

    # Remove λ=3 features
    removed = cae.get_removed()
    features_to_use = [f for f in all_features if f not in removed
                       and f in df.columns]

    X_df = df[features_to_use].copy()
    for col in X_df.columns:
        if X_df[col].dtype == object:
            X_df[col] = X_df[col].fillna(X_df[col].mode()[0])
        else:
            X_df[col] = X_df[col].fillna(X_df[col].median())

    X = X_df.values.astype(float)
    y = df['One_Year_Survival'].values.astype(int)

    print(f"Dataset: N={len(df)}, features={X.shape[1]}, "
          f"minority={y.sum()}, majority={(y==0).sum()}")
    print(f"CAE removed: {removed}")
    return X, y, features_to_use


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main(data_path: str, output_dir: str = 'results'):
    Path(output_dir).mkdir(exist_ok=True)
    X, y, feature_names = load_and_prepare(data_path)
    registry = get_model_registry()
    all_results = {}
    roc_data, pr_data = {}, {}

    for model_name, (factory, grid) in registry.items():
        print(f"  Running: {model_name} ...", end=' ', flush=True)
        y_true, y_prob = nested_cv(factory, grid, X, y)
        metrics = evaluate_model(y_true, y_prob)
        metrics['AUC_std'] = round(float(np.std([
            roc_auc_score(y_true[np.array([i for i in range(len(y_true))
                                            if i % 5 == f])],
                           y_prob[np.array([i for i in range(len(y_prob))
                                            if i % 5 == f])])
            for f in range(5)
            if len(np.unique(y_true[np.array([i for i in range(len(y_true))
                                               if i % 5 == f])])) > 1
        ])), 4)
        all_results[model_name] = metrics

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_data[model_name] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = round(average_precision_score(y_true, y_prob), 4)
        pr_data[model_name] = {'prec': prec.tolist(), 'rec': rec.tolist(), 'ap': ap}

        print(f"AUC={metrics['AUC']:.4f} [{metrics['AUC_CI_lo']}-{metrics['AUC_CI_hi']}]")

    # Save outputs
    results_df = pd.DataFrame(all_results).T
    results_df.to_csv(f'{output_dir}/model_performance.csv')
    with open(f'{output_dir}/roc_data.json', 'w') as f:
        json.dump(roc_data, f)
    with open(f'{output_dir}/pr_data.json', 'w') as f:
        json.dump(pr_data, f)

    print(f"\nResults saved to {output_dir}/")
    ranked = results_df.sort_values('AUC', ascending=False)
    print("\nModel ranking by AUC:")
    print(ranked[['AUC', 'AUC_CI_lo', 'AUC_CI_hi', 'Brier', 'HL_p']].to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CAE cardiac surgery ML pipeline')
    parser.add_argument('--data', default='data/synthetic_cardiac_cae_public.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--output', default='results', help='Output directory')
    args = parser.parse_args()
    main(args.data, args.output)
