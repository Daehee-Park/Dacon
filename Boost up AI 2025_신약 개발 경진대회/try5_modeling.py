# ------------------------- 1. Imports ---------------------------------------
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from custom_metrics import competition_scorer
import random
import os
import warnings
warnings.filterwarnings(action='ignore')
# ------------------------- 2. I/O -------------------------------------------
preprocessed_train_df = pd.read_csv('./output/try5/preprocessed_train.csv')
preprocessed_test_df  = pd.read_csv('./output/try5/preprocessed_test.csv')
test_df = pd.read_csv('./data/test.csv')
sample_submission_df = pd.read_csv('./data/sample_submission.csv')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(96)
# ------------------------- 3. Model -----------------------------------------
predictor = TabularPredictor(
    label='Inhibition', 
    problem_type='regression', 
    eval_metric=competition_scorer, 
    path='./output/try5/autogluon_models'
)

predictor.fit(train_data=preprocessed_train_df, presets='experimental_quality',time_limit=3600*6,full_weighted_ensemble_additionally=True,
            num_bag_folds=8,num_stack_levels=1,refit_full=True, num_cpus=8, memory_limit=16)

# ------------------------- 4. Predict ---------------------------------------
predictions = predictor.predict(preprocessed_test_df)

# ------------------------- 5. Save ------------------------------------------
submission = pd.DataFrame({
    'ID': sample_submission_df['ID'],
    'Inhibition': predictions
})
submission.to_csv('./output/try5/submission.csv', index=False)

predictor.leaderboard().to_csv('./output/try5/leaderboard.csv', index=False)

# ------------------------- 6. Submit ------------------------------------------
from dacon_submit import dacon_submit
leaderboard = predictor.leaderboard()
best_cv_score = leaderboard.iloc[0]['score_val']

dacon_submit(
    submission_path='./output/try5/submission.csv',
    memo=f'Try5-Bit Based Preprocessing with Autogluon, AutoGluon 10Hrs 5CV: {best_cv_score}'
)