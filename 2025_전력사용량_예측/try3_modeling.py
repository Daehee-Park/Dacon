import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from custom_metrics import smape
import optuna
from tqdm import tqdm

# 모델 import
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge

# 로깅 및 기본 설정
os.makedirs('output/try3', exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('output/try3/modeling.log', encoding='utf-8'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ------------------------- 1. 설정 및 데이터 로드 ------------------------------------
logger.info("========== 1. Loading Data and Configuration ==========")
TARGET = 'power_consumption'
N_SPLITS = 5
RANDOM_STATE = 42

train_df = pd.read_csv('./output/try3/preprocessed_train.csv')
test_df = pd.read_csv('./output/try3/preprocessed_test.csv')
submission_df = pd.read_csv('./data/sample_submission.csv')

X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_test = test_df.copy()

# ------------------------- 2. 기본 모델 및 교차 검증 -------------------------------
def get_base_models():
    """학습에 사용할 기본 모델들을 정의합니다."""
    return {
        'LGBM': lgb.LGBMRegressor(random_state=RANDOM_STATE),
        'XGB': xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        'CatBoost': cb.CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
        'RF': RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        'ET': ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    }

def train_predict_cv(model, X_train, y_train, X_test, model_name='model'):
    """교차 검증을 통해 OOF 예측과 테스트 예측을 생성합니다."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    oof_preds = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])
    
    for fold, (train_idx, val_idx) in enumerate(tqdm(kf.split(X_train), total=N_SPLITS, desc=f"CV for {model_name}")):
        X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        model.fit(X_train_fold, y_train_fold)
        oof_preds[val_idx] = model.predict(X_val_fold)
        test_preds += model.predict(X_test) / N_SPLITS
        
    score = smape(y_train, oof_preds)
    logger.info(f"Model: {model_name}, CV SMAPE: {score:.4f}")
    return oof_preds, test_preds, score

logger.info("========== 2. Training Base Models ==========")
base_models = get_base_models()
oof_preds = {}
test_preds = {}
scores = {}

for name, model in base_models.items():
    oof_preds[name], test_preds[name], scores[name] = train_predict_cv(model, X_train, y_train, X_test, model_name=name)

# ------------------------- 3. 모델 성능 평가 및 앙상블 ----------------------------
logger.info("========== 3. Evaluating Ensembles and Stacking Models ==========")
leaderboard = pd.DataFrame.from_dict(scores, orient='index', columns=['score']).sort_values('score')
logger.info(f"\n--- Base Model Leaderboard ---\n{leaderboard}")

# Top 5 모델 선정
top5_models = leaderboard.head(5).index.tolist()
logger.info(f"Top 5 Models: {top5_models}")

# 앙상블 모델 평가 (Simple Averaging)
oof_ensemble = np.mean([oof_preds[m] for m in top5_models], axis=0)
test_ensemble = np.mean([test_preds[m] for m in top5_models], axis=0)
ensemble_score = smape(y_train, oof_ensemble)
leaderboard.loc['Ensemble_Top5_Avg'] = ensemble_score
logger.info(f"Ensemble (Top 5 Avg) SMAPE: {ensemble_score:.4f}")

# 스태킹 모델 평가
# OOF 예측을 새로운 특성으로 사용
X_train_stack = pd.DataFrame({name: preds for name, preds in oof_preds.items()})[top5_models]
X_test_stack = pd.DataFrame({name: preds for name, preds in test_preds.items()})[top5_models]

stacking_meta_models = {
    'Stack_Ridge': Ridge(random_state=RANDOM_STATE),
    'Stack_LGBM': lgb.LGBMRegressor(random_state=RANDOM_STATE)
}

for name, meta_model in stacking_meta_models.items():
    oof, test, score = train_predict_cv(meta_model, X_train_stack, y_train, X_test_stack, model_name=name)
    leaderboard.loc[name] = score
    oof_preds[name], test_preds[name] = oof, test # 다음 단계를 위해 저장

# 최종 리더보드
leaderboard = leaderboard.sort_values('score')
logger.info(f"\n--- Full Leaderboard ---\n{leaderboard}")

# ------------------------- 4. HPO 및 최종 모델 학습 --------------------------
# (주의: HPO 과정은 매우 오래 걸릴 수 있습니다. n_trials를 작게 설정하여 테스트합니다.)
logger.info("========== 4. Hyperparameter Optimization ==========")
best_model_name = leaderboard.index[0]
logger.info(f"Best model found: {best_model_name}. Starting HPO for its components.")

# 이 예제에서는 시간 관계상 HPO를 생략하고, 최종 모델로 바로 예측을 진행합니다.
# 실제 사용 시 아래의 HPO 로직을 활성화하여 최적의 파라미터를 찾아야 합니다.
# 
# def objective(trial, model_name, X, y):
#     # ... Optuna 목적 함수 정의 ...
#     # ... 예: param = {'n_estimators': trial.suggest_int(...), ...} ...
#     # ... model = lgb.LGBMRegressor(**param) ...
#     # ... _, _, score = train_predict_cv(...) ...
#     # return score
#
# # study = optuna.create_study(direction='minimize')
# # study.optimize(lambda trial: objective(trial, 'LGBM', X_train, y_train), n_trials=50)
# # best_params_lgbm = study.best_params
# 
# # ... (다른 모든 컴포넌트 모델에 대해 HPO 반복) ...

logger.warning("Skipping HPO to save time. Using default parameters for final prediction.")
final_model_name = best_model_name

# 최종 모델로 예측 생성
if 'Stack' in final_model_name:
    logger.info(f"Training final stacking model: {final_model_name}")
    meta_model_final = stacking_meta_models[final_model_name]
    meta_model_final.fit(X_train_stack, y_train)
    final_predictions = meta_model_final.predict(X_test_stack)
elif 'Ensemble' in final_model_name:
    logger.info(f"Generating final predictions from ensemble: {final_model_name}")
    final_predictions = test_ensemble
else: # 기본 모델
    logger.info(f"Training final base model: {final_model_name}")
    final_model = get_base_models()[final_model_name]
    final_model.fit(X_train, y_train)
    final_predictions = final_model.predict(X_test)

final_predictions[final_predictions < 0] = 0

# ------------------------- 5. 제출 ------------------------------------------
logger.info("========== 5. Saving Submission File ==========")
submission_df['answer'] = final_predictions
submission_filepath = f'./output/try3/submission.csv'
submission_df.to_csv(submission_filepath, index=False)
logger.info(f"Submission file saved to {submission_filepath}")

from dacon_submit import dacon_submit
best_model_score = leaderboard.iloc[0]['score']
memo = f'Try3 (manual): Best model is {best_model_name}. Val SMAPE: {best_model_score:.4f}'
dacon_submit(
    submission_path=submission_filepath,
    memo=memo
)

logger.info("========== Modeling pipeline complete. ==========")