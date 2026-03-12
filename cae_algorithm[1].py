"""
cae_algorithm.py
----------------
Causal Adjacency Examination (CAE): a graph-based feature classification
procedure for supervised clinical ML pipelines.

Author: Dr. Ali Sadegh-Zadeh
        Staffordshire University

Reference:
    Sadegh-Zadeh, A. (2025). Causal Adjacency Examination (CAE): A Formalised
    Leakage-Detection Procedure for Clinical Machine Learning Pipelines.
    Artificial Intelligence Review [under review].

Algorithm
---------
Given:
    G = (V, E)  — a directed acyclic graph over clinical variables
    Y ∈ V       — the target node
    T0          — the prediction timepoint (e.g. discharge)

For each feature node X_i ∈ V \ {Y}:

    Q1: Does edge X_i → Y exist in G?
        Yes → λ = 3  (directly causal; remove)
        No  → proceed to Q2

    Q2: Is X_i observed at or before T0?
        Yes → λ = 1  (discharge-time; retain with annotation)
        No  → proceed to Q3

    Q3: (Reserved for future extension; currently unused in this implementation)
        → λ = 0  (pre-T0; retain unconditionally)

Complexity: O(d × |E|) time, O(|V| + |E|) space, where d = |V| − 1.
Deterministic: identical G and T0 always produce identical output.

Notes
-----
This module implements Q1 and Q2 only (Q3 reserved).
λ = 2 (indirect causal path through a mediator) is defined in the paper
but not present in the cardiac surgery dataset used here.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, Optional
import pandas as pd


# ─────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────

@dataclass
class CAEResult:
    """Output of the CAE procedure for a single feature."""
    feature: str
    lambda_score: int           # 0, 1, 2, or 3
    decision: str               # 'retain', 'retain_annotated', 'remove'
    q1_direct_cause: bool
    q2_available_at_t0: bool
    note: str = ""

    def __repr__(self):
        return (f"CAEResult(feature={self.feature!r}, λ={self.lambda_score}, "
                f"decision={self.decision!r})")


@dataclass
class CAEClassifier:
    """
    Implements the CAE procedure.

    Parameters
    ----------
    dag_edges : set of (str, str)
        Directed edges in the causal DAG, given as (cause, effect) pairs.
        Include all domain-knowledge edges, not only those involving Y.
    t0_features : set of str
        Features whose observation time is >= T0 (e.g. discharge-time variables).
        Features NOT in this set are assumed to be available before T0.
    target : str
        Name of the outcome variable Y.

    Example
    -------
    >>> dag = {('In_Hospital_Mortality', 'One_Year_Survival'),
    ...        ('ECMO', 'RV_Dysfunction')}
    >>> t0 = {'ECMO', 'CRRT', 'Furosemide', 'ICU_Time_Day'}
    >>> cae = CAEClassifier(dag_edges=dag, t0_features=t0,
    ...                     target='One_Year_Survival')
    >>> result = cae.classify('In_Hospital_Mortality')
    >>> result.lambda_score
    3
    """
    dag_edges: Set[tuple]
    t0_features: Set[str]
    target: str
    _results: Dict[str, CAEResult] = field(default_factory=dict, init=False)

    # ── internal helpers ───────────────────────────────────────────

    def _q1(self, feature: str) -> bool:
        """True if a direct edge feature → target exists in the DAG."""
        return (feature, self.target) in self.dag_edges

    def _q2(self, feature: str) -> bool:
        """True if the feature is available at or after T0."""
        return feature in self.t0_features

    # ── public interface ───────────────────────────────────────────

    def classify(self, feature: str) -> CAEResult:
        """Classify a single feature and return a CAEResult."""
        if feature == self.target:
            raise ValueError(f"Cannot classify the target variable itself: {feature!r}")

        q1 = self._q1(feature)
        q2 = self._q2(feature)

        if q1:
            result = CAEResult(
                feature=feature,
                lambda_score=3,
                decision='remove',
                q1_direct_cause=True,
                q2_available_at_t0=q2,
                note="Direct cause of outcome; removed to prevent outcome leakage."
            )
        elif q2:
            result = CAEResult(
                feature=feature,
                lambda_score=1,
                decision='retain_annotated',
                q1_direct_cause=False,
                q2_available_at_t0=True,
                note="Available at T0; retained with sensitivity annotation."
            )
        else:
            result = CAEResult(
                feature=feature,
                lambda_score=0,
                decision='retain',
                q1_direct_cause=False,
                q2_available_at_t0=False,
                note="Pre-T0 feature; retained unconditionally."
            )

        self._results[feature] = result
        return result

    def classify_all(self, features: list[str]) -> Dict[str, CAEResult]:
        """Classify a list of features. Returns dict keyed by feature name."""
        for f in features:
            self.classify(f)
        return dict(self._results)

    def get_retained(self, include_annotated: bool = True) -> list[str]:
        """Return list of features to retain after CAE filtering."""
        keep = []
        for f, r in self._results.items():
            if r.decision == 'retain':
                keep.append(f)
            elif r.decision == 'retain_annotated' and include_annotated:
                keep.append(f)
        return keep

    def get_removed(self) -> list[str]:
        """Return list of features removed by CAE (λ=3)."""
        return [f for f, r in self._results.items() if r.decision == 'remove']

    def summary_dataframe(self) -> pd.DataFrame:
        """Return a summary DataFrame suitable for Table 1 in a manuscript."""
        rows = []
        for f, r in self._results.items():
            rows.append({
                'Feature': f,
                'lambda': r.lambda_score,
                'Decision': r.decision,
                'Q1_DirectCause': r.q1_direct_cause,
                'Q2_AtT0': r.q2_available_at_t0,
                'Note': r.note,
            })
        df = pd.DataFrame(rows).sort_values('lambda')
        return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# Cardiac surgery DAG — as specified in the manuscript
# ─────────────────────────────────────────────────────────────────

def build_cardiac_surgery_cae() -> CAEClassifier:
    """
    Instantiate the CAE classifier for the cardiac surgery cohort.

    DAG edges and T0 assignments are based on clinical domain knowledge
    (Dr. Ali Sadegh-Zadeh, Staffordshire University) and are documented
    in the manuscript Table 1 and Supplementary Material.

    T0 = discharge from index admission.
    """
    # Direct causal edges X → One_Year_Survival in the domain DAG.
    # Only In_Hospital_Mortality has a direct causal edge; all other
    # discharge-time variables are prognostically associated but not
    # direct causes (they are indicators of severity, not causes of death
    # in the DAG sense used here).
    dag_edges = {
        ('In_Hospital_Mortality', 'One_Year_Survival'),
    }

    # Features first observed at or after T0 = discharge.
    # These are λ=1: available for a discharge-risk model but
    # not for a pre-operative risk model.
    t0_features = {
        'ECMO', 'CRRT', 'Furosemide', 'Prostaglandins',
        'SildenafilTadalafil', 'AntiCoagulation_Use',
        'Mil_Time_D', 'Sil_Time_D', 'PG_Time_D',
        'Intubation_Time_Hour', 'ICU_Time_Day', 'Pump_Time_Minute',
        'NEN_After_Tx', 'NEN_Time_D', 'Mil_After_Tx',
        'Rejection_In_First_Week',
        'CVP_Third_Day', 'CVP_Third_Day_Binary', 'RV_Dysfunction',
    }

    return CAEClassifier(
        dag_edges=dag_edges,
        t0_features=t0_features,
        target='One_Year_Survival'
    )


# ─────────────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    cae = build_cardiac_surgery_cae()

    test_features = [
        'In_Hospital_Mortality',   # expect λ=3
        'ECMO',                    # expect λ=1
        'Age',                     # expect λ=0
        'Albumin',                 # expect λ=0
        'Furosemide',              # expect λ=1
    ]

    print("CAE self-test — cardiac surgery cohort")
    print(f"{'Feature':<30} {'λ':<4} {'Decision'}")
    print("-" * 55)
    for feat in test_features:
        r = cae.classify(feat)
        print(f"{r.feature:<30} {r.lambda_score:<4} {r.decision}")
