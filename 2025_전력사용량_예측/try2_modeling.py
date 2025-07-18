# ------------------------- 1. Imports ---------------------------------------
import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from custom_metrics import ag_smape_scorer

# ------------------------- 2. I/O -------------------------------------------
preprocessed_train_df = pd.read_csv('./output/try2/preprocessed_train.csv')
preprocessed_test_df  = pd.read_csv('./output/try2/preprocessed_test.csv')
test_df = pd.read_csv('./data/test.csv')

# ------------------------- 3. Model -----------------------------------------
predictor = TabularPredictor(
    label='power_consumption', 
    problem_type='regression', 
    eval_metric=ag_smape_scorer,
    path='./output/try2/autogluon_models'
)

predictor.fit(train_data=preprocessed_train_df, presets='best_quality',time_limit=3600*10,full_weighted_ensemble_additionally=True,dynamic_stacking=True,
              fit_strategy="parallel", auto_stack=True,num_bag_folds=5,num_stack_levels=1, ds_args={'enable_ray_logging':False, 'n_folds':5}, num_cpus=18)

# ------------------------- 4. Predict ---------------------------------------
predictions = predictor.predict(preprocessed_test_df)

# ------------------------- 5. Save ------------------------------------------
submission = pd.DataFrame({
    'num_date_time': test_df['num_date_time'],
    'answer': predictions
})
submission.to_csv('./output/try2/submission.csv', index=False)

predictor.leaderboard().to_csv('./output/try2/leaderboard.csv', index=False)

# ------------------------- 6. Submit ------------------------------------------
from dacon_submit import dacon_submit
leaderboard = predictor.leaderboard()
best_cv_score = leaderboard.iloc[0]['score_val']

dacon_submit(
    submission_path='./output/try2/submission.csv',
    memo=f'Try2: Predicted sunshine/solar_radiation. Enhanced skew transforms. Added temp*humidity feature. AutoGluon 5CV: {best_cv_score}'
)