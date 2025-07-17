import pandas as pd
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor.load('./output/try2/autogluon_models/')
print(predictor.leaderboard().to_csv('./output/try2/leaderboard.csv', index=False))