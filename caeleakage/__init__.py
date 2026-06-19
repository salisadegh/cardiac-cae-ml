"""
caeleakage — CAE leakage-risk classification for clinical ML pipelines.

Install:
    pip install git+https://github.com/salisadegh/caeleakage.git

Quick start:
    from caeleakage import CAEClassifier, CAEPipeline

    cae = CAEClassifier(
        dag_edges   = {('In_Hospital_Mortality', 'One_Year_Survival')},
        t0_features = {'ECMO', 'CRRT', 'Furosemide'},
        target      = 'One_Year_Survival'
    )
    cae.fit(X_df); cae.report()

    pipeline = CAEPipeline(cae_classifier=cae)
    results  = pipeline.fit_evaluate(X_df, y)
    print(results['AUC'], results['AUC_CI'])
"""

from .classifier import CAEClassifier, CAEResult
from .pipeline   import CAEPipeline, smote_oversample

__version__ = '0.1.0'
__author__  = 'Ali Sadegh-Zadeh'
__email__   = 'ali.sadegh-zadeh@staffs.ac.uk'
__license__ = 'MIT'

__all__ = ['CAEClassifier', 'CAEResult', 'CAEPipeline', 'smote_oversample']
