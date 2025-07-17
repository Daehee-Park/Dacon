# ------------------------- 1. Imports ---------------------------------------
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from custom_metrics import competition_scorer
# ------------------------- 2. I/O -------------------------------------------
preprocessed_train_df = pd.read_csv('./output/try3/preprocessed_train.csv')
preprocessed_test_df  = pd.read_csv('./output/try3/preprocessed_test.csv')
# ------------------------- 3. Model -----------------------------------------
predictor = TabularPredictor(label='Inhibition', problem_type='regression', eval_metric=competition_scorer, path='./output/try3/autogluon_models')
predictor.fit(train_data=preprocessed_train_df, presets='best_quality',time_limit=3600*2,full_weighted_ensemble_additionally=True,dynamic_stacking=True,
              fit_strategy="parallel", auto_stack=True,num_bag_folds=5,num_stack_levels=1)
# ------------------------- 4. Predict ---------------------------------------
predictions = predictor.predict(preprocessed_test_df)
# ------------------------- 5. Save ------------------------------------------
test_df = pd.read_csv('./data/test.csv')
test_df['Inhibition'] = predictions
test_df.drop(columns=['Canonical_Smiles'], inplace=True)
test_df.to_csv('./output/try3/submission.csv', index=False)
predictor.leaderboard().to_csv('./output/try3/leaderboard.csv', index=False)
# ------------------------- 6. Submit ------------------------------------------
from dacon_submit import dacon_submit
leaderboard = predictor.leaderboard()
best_cv_score = leaderboard.iloc[0]['score_val']
dacon_submit(submission_path='./output/try3/submission.csv', memo=f'Try3-Count Based Preprocessing+MACCS+AtomPair+RF Importance Selection with Autogluon, 5CV: {best_cv_score}')