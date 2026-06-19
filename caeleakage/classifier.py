"""
caeleakage.classifier
---------------------
Causal Adjacency Examination (CAE) — core feature classification algorithm.

Reference:
    Sadegh-Zadeh et al. (2025). CAE: A Reproducible, DAG-Guided
    Leakage-Risk Classification Procedure for Clinical ML Pipelines.
    Computer Methods and Programs in Biomedicine [under review].

GitHub: https://github.com/salisadegh/caeleakage
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
import warnings


@dataclass
class CAEResult:
    """CAE classification result for a single feature.

    Attributes
    ----------
    feature : str
    lambda_score : int  — 0, 1, 2, or 3
    decision : str      — 'retain' | 'retain_annotated' | 'flag' | 'remove'
    q1_direct_cause : bool
    q2_deployment_dependent : bool
    q3_proxy : bool
    reason : str
    """
    feature: str
    lambda_score: int
    decision: str
    q1_direct_cause: bool
    q2_deployment_dependent: bool
    q3_proxy: bool = False
    reason: str = ""

    @property
    def is_leaky(self) -> bool:
        return self.lambda_score == 3

    @property
    def needs_sensitivity(self) -> bool:
        return self.lambda_score in (1, 2)

    def __repr__(self) -> str:
        return f"CAEResult(feature={self.feature!r}, λ={self.lambda_score}, decision={self.decision!r})"


class CAEClassifier:
    """
    Causal Adjacency Examination (CAE) procedure.

    Assigns each feature a leakage-risk score λ ∈ {0, 1, 2, 3}
    via three sequential questions grounded in the causal DAG:

    Q1 — Direct causal adjacency: does edge X_i → Y exist in G?
         YES → λ=3, remove unconditionally.

    Q2 — Deployment-dependent: is X_i first observed at or after T₀?
         YES → λ=1, retain with sensitivity annotation.
         (Note: λ=1 features are NOT unconditionally leaky; they are
          legitimate predictors when the model is deployed at T₀.)

    Q3 — Proxy: is X_i in the user-specified proxy set P?
         YES → λ=2, flag for expert adjudication.

    Default → λ=0, retain unconditionally.

    Complexity: O(d × |E|). Deterministic given G, T₀, and P.

    Parameters
    ----------
    dag_edges : set of (str, str)
        Directed edges (cause, effect) in the causal DAG.
    t0_features : set of str
        Features whose first observation time is at or after T₀
        (deployment-dependent features, λ=1).
    proxy_features : set of str, optional
        User-specified indirect proxy features (λ=2).
    target : str
        Name of the outcome variable Y.

    Examples
    --------
    >>> cae = CAEClassifier(
    ...     dag_edges={('In_Hospital_Mortality', 'One_Year_Survival')},
    ...     t0_features={'ECMO', 'CRRT', 'Furosemide'},
    ...     target='One_Year_Survival'
    ... )
    >>> cae.classify('In_Hospital_Mortality').lambda_score
    3
    >>> cae.classify('Age').lambda_score
    0
    """

    def __init__(
        self,
        dag_edges: Optional[Set[Tuple[str, str]]] = None,
        t0_features: Optional[Set[str]] = None,
        proxy_features: Optional[Set[str]] = None,
        target: str = 'outcome',
    ):
        self.dag_edges      = set(dag_edges)      if dag_edges      else set()
        self.t0_features    = set(t0_features)    if t0_features    else set()
        self.proxy_features = set(proxy_features) if proxy_features else set()
        self.target         = target
        self._results: Dict[str, CAEResult] = {}

    def _q1(self, feature: str) -> bool:
        return (feature, self.target) in self.dag_edges

    def _q2(self, feature: str) -> bool:
        return feature in self.t0_features

    def _q3(self, feature: str) -> bool:
        return feature in self.proxy_features

    def classify(self, feature: str) -> CAEResult:
        """Classify a single feature. Returns CAEResult."""
        if feature == self.target:
            raise ValueError(f"Cannot classify the target variable: {feature!r}")

        q1 = self._q1(feature)
        q2 = self._q2(feature)
        q3 = self._q3(feature)

        if q1:
            result = CAEResult(
                feature=feature, lambda_score=3, decision='remove',
                q1_direct_cause=True, q2_deployment_dependent=q2,
                reason=f"Direct edge {feature!r}→{self.target!r} in DAG. Remove unconditionally."
            )
        elif q2:
            result = CAEResult(
                feature=feature, lambda_score=1, decision='retain_annotated',
                q1_direct_cause=False, q2_deployment_dependent=True,
                reason=f"{feature!r} is deployment-dependent (observed at T₀). Retain in Tier-1; exclude in Tier-2."
            )
        elif q3:
            result = CAEResult(
                feature=feature, lambda_score=2, decision='flag',
                q1_direct_cause=False, q2_deployment_dependent=False, q3_proxy=True,
                reason=f"{feature!r} is an indirect proxy. Flag for expert adjudication."
            )
        else:
            result = CAEResult(
                feature=feature, lambda_score=0, decision='retain',
                q1_direct_cause=False, q2_deployment_dependent=False,
                reason=f"{feature!r} is a safe baseline feature. Retain unconditionally."
            )

        self._results[feature] = result
        return result

    def classify_all(self, features: List[str]) -> Dict[str, CAEResult]:
        """Classify a list of features."""
        for f in features:
            self.classify(f)
        return dict(self._results)

    def fit(self, X: pd.DataFrame) -> 'CAEClassifier':
        """Classify all columns in a DataFrame (excluding target if present)."""
        features = [c for c in X.columns if c != self.target]
        self.classify_all(features)
        return self

    def transform(self, X: pd.DataFrame, include_annotated: bool = True) -> pd.DataFrame:
        """Remove λ=3 features; optionally remove λ=1/2 (conservative Tier-2)."""
        if not self._results:
            warnings.warn("Call fit() first.", UserWarning)
            return X
        drop = self.removed_features_
        if not include_annotated:
            drop = drop + self.annotated_features_
        return X[[c for c in X.columns if c not in drop]]

    def fit_transform(self, X: pd.DataFrame, include_annotated: bool = True) -> pd.DataFrame:
        """Fit then transform."""
        return self.fit(X).transform(X, include_annotated=include_annotated)

    @property
    def retained_features_(self) -> List[str]:
        return [f for f, r in self._results.items() if r.lambda_score == 0]

    @property
    def annotated_features_(self) -> List[str]:
        return [f for f, r in self._results.items() if r.lambda_score in (1, 2)]

    @property
    def removed_features_(self) -> List[str]:
        return [f for f, r in self._results.items() if r.lambda_score == 3]

    def summary(self) -> pd.DataFrame:
        """Return a summary DataFrame of all classified features."""
        rows = [{'Feature': f, 'lambda': r.lambda_score, 'decision': r.decision,
                 'Q1_direct': r.q1_direct_cause, 'Q2_dep_dep': r.q2_deployment_dependent,
                 'Q3_proxy': r.q3_proxy, 'reason': r.reason}
                for f, r in self._results.items()]
        return pd.DataFrame(rows).sort_values('lambda').reset_index(drop=True)

    def report(self) -> str:
        """Print a human-readable CAE classification report."""
        if not self._results:
            return "CAEClassifier: no features classified yet."
        lines = [
            "=" * 65,
            "CAE CLASSIFICATION REPORT",
            f"Target: {self.target}",
            "=" * 65,
            f"{'Feature':<35} {'λ':<4} {'Decision'}",
            "-" * 65,
        ]
        for _, r in sorted(self._results.items(), key=lambda x: x[1].lambda_score):
            lines.append(f"{r.feature:<35} {r.lambda_score:<4} {r.decision}")
        lines += [
            "-" * 65,
            f"λ=0 retain:              {len(self.retained_features_):>3}",
            f"λ=1/2 annotated:         {len(self.annotated_features_):>3}",
            f"λ=3 removed (leaky):     {len(self.removed_features_):>3}",
            "=" * 65,
        ]
        report_str = "\n".join(lines)
        print(report_str)
        return report_str

    def __repr__(self) -> str:
        return (f"CAEClassifier(target={self.target!r}, "
                f"classified={len(self._results)}, removed={len(self.removed_features_)})")
