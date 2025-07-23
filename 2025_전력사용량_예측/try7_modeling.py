# ------------------------- 1. Imports ---------------------------------------
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from custom_metrics import ag_smape_scorer
import random
import os
import warnings
warnings.filterwarnings(action='ignore')
# ------------------------- 2. I/O -------------------------------------------
preprocessed_train_df = pd.read_csv('./output/try7/preprocessed_train.csv')
preprocessed_test_df  = pd.read_csv('./output/try7/preprocessed_test.csv')
test_df = pd.read_csv('./data/test.csv')
sample_submission_df = pd.read_csv('./data/sample_submission.csv')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(77)
# ------------------------- 3. Model -----------------------------------------
predictor = TabularPredictor(
    label='power_consumption', 
    problem_type='regression', 
    eval_metric=ag_smape_scorer, 
    path='./output/try7/autogluon_models'
)

predictor.fit(train_data=preprocessed_train_df, presets='experimental_quality',time_limit=3600*4,full_weighted_ensemble_additionally=True,
            num_bag_folds=10,num_stack_levels=1,refit_full=True,num_cpus=24,memory_limit=96)

# ------------------------- 4. Predict ---------------------------------------
predictions = predictor.predict(preprocessed_test_df)

# ------------------------- 5. Save ------------------------------------------
submission = pd.DataFrame({
    'num_date_time': sample_submission_df['num_date_time'],
    'answer': predictions
})
submission.to_csv('./output/try7/submission.csv', index=False)

predictor.leaderboard().to_csv('./output/try7/leaderboard.csv', index=False)

# ------------------------- 6. Submit ------------------------------------------
from dacon_submit import dacon_submit
leaderboard = predictor.leaderboard()
best_cv_score = leaderboard.iloc[0]['score_val']

dacon_submit(
    submission_path='./output/try7/submission.csv',
    memo=f'Try7: Advanced FE (cyclical, holiday, DI, lag), LGBM imputation for sun features, Skewness handled, AutoGluon 4Hrs 10CV: {best_cv_score}'
)