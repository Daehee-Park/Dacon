from autogluon.tabular import TabularPredictor
import pandas as pd

model = TabularPredictor.load("output/try3/autogluon_models")
leaderboard = model.leaderboard(extra_info=True)
leaderboard.to_csv("output/try3/leaderboard.csv", index=False)