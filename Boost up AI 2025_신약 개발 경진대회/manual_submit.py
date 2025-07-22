import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from custom_metrics import competition_scorer
sample_submission_df = pd.read_csv('./data/sample_submission.csv')
preprocessed_test_df  = pd.read_csv('./output/try5/preprocessed_test.csv')
predictor = TabularPredictor.load('./output/try5/autogluon_models/')
predictions = predictor.predict(preprocessed_test_df)
submission = pd.DataFrame({
    'ID': sample_submission_df['ID'],
    'Inhibition': predictions
})
submission.to_csv('./output/try5/submission.csv', index=False)
predictor.leaderboard().to_csv('./output/try5/leaderboard.csv', index=False)

from dacon_submit import dacon_submit
leaderboard = predictor.leaderboard()
best_cv_score = leaderboard.iloc[0]['score_val']

dacon_submit(
    submission_path='./output/try5/submission.csv',
    memo=f'Try5-Bit Based Preprocessing with Autogluon, AutoGluon 6Hrs 5CV: {best_cv_score}'
)