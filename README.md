# cardiac-cae-ml

Analysis code, figures, and results for:

> Sadegh-Zadeh et al. (2025). *CAE: A Reproducible, DAG-Guided Leakage-Risk
> Classification Procedure for Clinical ML Pipelines.*
> Computer Methods and Programs in Biomedicine (under review).

---

## Two repositories

| Repository | Purpose |
|---|---|
| **[cardiac-cae-ml](https://github.com/salisadegh/cardiac-cae-ml)** | This repo — paper analysis code |
| **[caeleakage](https://github.com/salisadegh/caeleakage)** | Reusable Python package |

---

## Repository structure

```
cardiac-cae-ml/
├── caeleakage/                  # Local copy of package (same as pip install)
│   ├── __init__.py
│   ├── classifier.py
│   └── pipeline.py
├── experiments/
│   ├── 01_uci_heart_failure.py  # Figs 2–3, Table 2 (UCI rows)
│   ├── 02_nhanes_specificity.py # Table 2 (NHANES row), ablation
│   └── 03_simulation_study.py   # Table 1 (all simulation scenarios)
├── figures/
│   ├── figure1_forest_cardiac.png
│   ├── figure2_uci_naive_vs_cae.png
│   ├── figure3_uci_inflation.png
│   ├── figure4_nhanes_mortality_rates.png
│   └── figure5_cross_domain_summary.png
├── results/
│   ├── simulation_main_results.json
│   ├── simulation_extra_scenarios.json
│   └── nhanes_ablation_results.json
├── data/
│   └── synthetic_cardiac_N300.csv  ← privacy-safe (no patient data)
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/salisadegh/cardiac-cae-ml.git
cd cardiac-cae-ml
pip install -r requirements.txt
```

## Reproduce results

```bash
cd experiments/
python 01_uci_heart_failure.py     # Table 2 UCI + Figs 2–3
python 02_nhanes_specificity.py    # Table 2 NHANES + ablation
python 03_simulation_study.py      # Table 1 (all 4 scenarios)
```

---

## Data availability

| Dataset | Source | Included? |
|---|---|---|
| UCI Heart Failure | [UCI ML Repository](https://archive.ics.uci.edu/dataset/519/) | Download separately |
| NHANES 2017–2018 | [CDC NHANES](https://wwwn.cdc.gov/nchs/nhanes/) | Download separately |
| NHANES Mortality | [NCHS Linked Mortality](https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/) | Download separately |
| Cardiac surgery | Ethics: IR.UMSU.REC.1403.198 — contact corresponding author | ✗ (DUA required) |
| **Synthetic (N=300)** | Gaussian copula | **✅ included** |

---

## Citation

```bibtex
@article{sadeghzadeh2025cae,
  title  = {CAE: A Reproducible, DAG-Guided Leakage-Risk Classification Procedure
             for Clinical Machine Learning Pipelines},
  author = {Sadegh-Zadeh, Seyed-Ali and others},
  journal= {Computer Methods and Programs in Biomedicine},
  year   = {2025},
  note   = {Under review}
}
```

## License

MIT — see [LICENSE](LICENSE).
