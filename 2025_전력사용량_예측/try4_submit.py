# ------------------------- 1. Imports ---------------------------------------
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from custom_metrics import ag_smape_scorer

predictor = TabularPredictor.load('./output/try4/autogluon_models/')
from dacon_submit import dacon_submit
leaderboard = predictor.leaderboard()
best_cv_score = leaderboard.iloc[0]['score_val']

dacon_submit(
    submission_path='./output/try4/submission.csv',
    memo=f'Regression: Imputed sunshine/solar for test, Feature engineering, log transform, building-hour avg features, AutoGluon 5CV: {best_cv_score}'
)