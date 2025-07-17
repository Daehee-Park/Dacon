# ------------------------- 1. Imports ---------------------------------------
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from custom_metrics import competition_scorer
# ------------------------- 2. I/O -------------------------------------------
preprocessed_train_df = pd.read_csv('./output/try2/preprocessed_train.csv')
preprocessed_test_df  = pd.read_csv('./output/try2/preprocessed_test.csv')
# ------------------------- 3. Model -----------------------------------------
predictor = TabularPredictor(label='Inhibition', problem_type='regression', eval_metric=competition_scorer, path='./output/try2/autogluon_models')
predictor.fit(train_data=preprocessed_train_df, presets='best_quality',time_limit=3600*2,full_weighted_ensemble_additionally=True,dynamic_stacking=True,
              fit_strategy="parallel", auto_stack=True,num_bag_folds=5)
# ------------------------- 4. Predict ---------------------------------------
predictions = predictor.predict(preprocessed_test_df)
# ------------------------- 5. Save ------------------------------------------
test_df = pd.read_csv('./data/test.csv')
test_df['Inhibition'] = predictions
test_df.drop(columns=['Canonical_Smiles'], inplace=True)
test_df.to_csv('./output/try2/submission.csv', index=False)