"""
caeleakage.pipeline
-------------------
Corrected nested CV pipeline with SMOTE-inside-fold.

CRITICAL: SMOTE is applied ONLY inside each inner-training fold.
Applying SMOTE before CV splits inflates AUC by 1.7–3.3 points.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Optional, Tuple, Union

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.svm import SVC

from .classifier import CAEClassifier


def smote_oversample(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    SMOTE — call ONLY inside inner training fold.

    WARNING: Do NOT apply before cross-validation splits.
    Doing so contaminates inner-validation folds and inflates AUC
    by 1.7–3.3 points (Sadegh-Zadeh et al., 2025, Section 2.2).
    """
    rng = np.random.RandomState(random_state)
    X, y = np.array(X, dtype=float), np.array(y)
    classes, counts = np.unique(y, return_counts=True)
    minority = classes[np.argmin(counts)]
    X_min = X[y == minority]
    n_syn = counts.max() - counts.min()
    synthetic = []
    for _ in range(n_syn):
        idx = rng.randint(0, len(X_min))
        sample = X_min[idx]
        dists = np.sqrt(((X_min - sample) ** 2).sum(axis=1))
        dists[idx] = np.inf
        k_eff = min(k, len(X_min) - 1)
        nb = X_min[np.argsort(dists)[:k_eff]][rng.randint(0, k_eff)]
        synthetic.append(sample + rng.rand() * (nb - sample))
    return np.vstack([X, np.array(synthetic)]), np.concatenate([y, np.full(n_syn, minority)])


class CAEPipeline:
    """
    Corrected clinical ML pipeline: CAE + nested CV + SMOTE-inside-fold.

    Parameters
    ----------
    estimator : sklearn estimator
        Defaults to SVM (Linear, balanced).
    cae_classifier : CAEClassifier, optional
        Pre-fitted CAEClassifier. If None, no CAE filtering is applied.
    n_outer : int   — outer CV folds (default: 5)
    n_inner : int   — inner CV folds (default: 3)
    param_grid : list of dict — hyperparameter search grid
    apply_smote : bool — apply SMOTE inside inner training folds (default: True)
    smote_k : int   — SMOTE nearest neighbours (default: 3)
    n_bootstrap : int — bootstrap replicates for 95% CI (default: 2000)
    random_state : int
    include_annotated : bool — include λ=1 features in Tier-1 (default: True)
    verbose : bool

    Examples
    --------
    >>> pipeline = CAEPipeline(cae_classifier=cae, n_outer=5, n_inner=3)
    >>> results = pipeline.fit_evaluate(X_df, y)
    >>> print(results['AUC'], results['AUC_CI'])
    """

    def __init__(
        self,
        estimator=None,
        cae_classifier: Optional[CAEClassifier] = None,
        n_outer: int = 5,
        n_inner: int = 3,
        param_grid: Optional[List[dict]] = None,
        apply_smote: bool = True,
        smote_k: int = 3,
        n_bootstrap: int = 2000,
        random_state: int = 42,
        include_annotated: bool = True,
        verbose: bool = True,
    ):
        if estimator is None:
            estimator = SVC(
                kernel='linear', C=1.0, probability=True,
                class_weight='balanced', random_state=random_state
            )
        self.estimator       = estimator
        self.cae_classifier  = cae_classifier
        self.n_outer         = n_outer
        self.n_inner         = n_inner
        self.param_grid      = param_grid or [{}]
        self.apply_smote     = apply_smote
        self.smote_k         = smote_k
        self.n_bootstrap     = n_bootstrap
        self.random_state    = random_state
        self.include_annotated = include_annotated
        self.verbose         = verbose
        self._y_true = None
        self._y_prob = None

    def _get_prob(self, clf, X):
        if hasattr(clf, 'predict_proba'):
            return clf.predict_proba(X)[:, 1]
        d = clf.decision_function(X)
        return (d - d.min()) / (d.max() - d.min() + 1e-8)

    def _inner_loop(self, X_tr, y_tr, fold_seed):
        inner_cv = StratifiedKFold(
            n_splits=self.n_inner, shuffle=True, random_state=fold_seed
        )
        best_auc, best_params = -1, self.param_grid[0]
        for params in self.param_grid:
            aucs = []
            for itr, ival in inner_cv.split(X_tr, y_tr):
                Xi, yi = X_tr[itr], y_tr[itr]
                if self.apply_smote and yi.sum() >= 3 and (yi == 0).sum() >= 3:
                    k = min(self.smote_k, max(1, int((yi == 0).sum()) - 1))
                    Xi, yi = smote_oversample(Xi, yi, k=k, random_state=fold_seed)
                sc = StandardScaler()
                clf = clone(self.estimator)
                try:
                    clf.set_params(**params)
                except Exception:
                    pass
                clf.fit(sc.fit_transform(Xi), yi)
                if len(np.unique(y_tr[ival])) > 1:
                    aucs.append(roc_auc_score(y_tr[ival], self._get_prob(clf, sc.transform(X_tr[ival]))))
            if aucs and np.mean(aucs) > best_auc:
                best_auc = np.mean(aucs)
                best_params = params
        return best_params

    def fit_evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: np.ndarray,
    ) -> Dict:
        """
        Run full CAE + nested CV pipeline.

        Returns dict with: AUC, AUC_CI, AUC_CI_lo, AUC_CI_hi,
                           Brier, y_true, y_prob,
                           features_used, features_removed
        """
        features_removed, features_used = [], None

        if self.cae_classifier is not None:
            if isinstance(X, pd.DataFrame):
                if not self.cae_classifier._results:
                    self.cae_classifier.fit(X)
                X_filtered = self.cae_classifier.transform(
                    X, include_annotated=self.include_annotated
                )
                features_removed = self.cae_classifier.removed_features_
                features_used = list(X_filtered.columns)
                X_arr = X_filtered.values.astype(float)
            else:
                warnings.warn("Pass a DataFrame for CAE filtering.", UserWarning)
                X_arr = np.array(X, dtype=float)
        else:
            X_arr = np.array(X, dtype=float) if not isinstance(X, np.ndarray) else X.astype(float)

        y_arr = np.array(y)
        outer_cv = StratifiedKFold(
            n_splits=self.n_outer, shuffle=True, random_state=self.random_state
        )
        y_true_all, y_prob_all = [], []

        for fold_idx, (tr, te) in enumerate(outer_cv.split(X_arr, y_arr)):
            X_tr, X_te = X_arr[tr], X_arr[te]
            y_tr, y_te = y_arr[tr], y_arr[te]
            best_params = self._inner_loop(X_tr, y_tr, fold_seed=fold_idx)

            if self.apply_smote and y_tr.sum() >= 3 and (y_tr == 0).sum() >= 3:
                k = min(self.smote_k, max(1, int((y_tr == 0).sum()) - 1))
                X_sm, y_sm = smote_oversample(X_tr, y_tr, k=k, random_state=fold_idx)
            else:
                X_sm, y_sm = X_tr, y_tr

            sc = StandardScaler()
            clf = clone(self.estimator)
            try:
                clf.set_params(**best_params)
            except Exception:
                pass
            clf.fit(sc.fit_transform(X_sm), y_sm)
            y_prob = self._get_prob(clf, sc.transform(X_te))
            y_true_all.extend(y_te)
            y_prob_all.extend(y_prob)

            if self.verbose:
                fold_auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else float('nan')
                print(f"  Fold {fold_idx+1}/{self.n_outer}  AUC={fold_auc:.4f}")

        self._y_true = np.array(y_true_all)
        self._y_prob = np.array(y_prob_all)

        auc   = round(roc_auc_score(self._y_true, self._y_prob), 4)
        brier = round(brier_score_loss(self._y_true, self._y_prob), 4)
        ci_lo, ci_hi = self._bootstrap_ci()

        if self.verbose:
            print(f"\n  AUC = {auc:.4f}  95% CI [{ci_lo}–{ci_hi}]  Brier = {brier:.4f}")
            if features_removed:
                print(f"  CAE removed (λ=3): {features_removed}")

        return {
            'AUC': auc, 'AUC_CI': (ci_lo, ci_hi),
            'AUC_CI_lo': ci_lo, 'AUC_CI_hi': ci_hi,
            'Brier': brier,
            'y_true': self._y_true, 'y_prob': self._y_prob,
            'features_used': features_used, 'features_removed': features_removed,
        }

    def _bootstrap_ci(self, alpha: float = 0.05) -> Tuple[float, float]:
        rng = np.random.RandomState(self.random_state)
        scores = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(len(self._y_true), len(self._y_true), replace=True)
            if len(np.unique(self._y_true[idx])) < 2:
                continue
            scores.append(roc_auc_score(self._y_true[idx], self._y_prob[idx]))
        return (round(np.percentile(scores, 100 * alpha / 2), 4),
                round(np.percentile(scores, 100 * (1 - alpha / 2)), 4))
