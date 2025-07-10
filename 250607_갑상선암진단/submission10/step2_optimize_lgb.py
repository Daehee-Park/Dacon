import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
import json
import os

warnings.filterwarnings('ignore')

RESULT_PATH = './result/'

print("=== Step 2: LightGBM Optimization ===")

# 전처리된 데이터 로드
if not os.path.exists(RESULT_PATH + 'X_train.csv'):
    print("❌ 전처리된 데이터가 없습니다. step1_preprocess.py를 먼저 실행하세요.")
    exit()

X_train = pd.read_csv(RESULT_PATH + 'X_train.csv')
y_train = pd.read_csv(RESULT_PATH + 'y_train.csv')['Cancer']

print(f"데이터 로드 완료: {X_train.shape}")
print(f"Target 분포: {y_train.value_counts().to_dict()}")

# CV 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 9차 시도 최적 파라미터 참조 (범위 설정용)
# 기존 최적: learning_rate=0.012, n_estimators=930, num_leaves=89, etc.
reference_params = {
    'n_estimators': 930,
    'learning_rate': 0.01293,
    'num_leaves': 89,
    'max_depth': 5,
    'feature_fraction': 0.7017,
    'bagging_fraction': 0.8293,
    'min_child_samples': 49,
    'reg_alpha': 2.6747,
    'reg_lambda': 5.4030
}

print(f"\n참조 파라미터:")
for key, value in reference_params.items():
    print(f"- {key}: {value}")

# F1 최적화 함수
def optimize_threshold(y_true, y_prob):
    """최적 threshold 찾기"""
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in np.arange(0.1, 0.7, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

def evaluate_lgb_cv(params, X, y, cv_folds=5):
    """LightGBM CV 평가"""
    f1_scores = []
    thresholds = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 모델 학습
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr)
        
        # 예측 확률
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # 최적 threshold 및 F1 score
        threshold, f1 = optimize_threshold(y_val, y_prob)
        f1_scores.append(f1)
        thresholds.append(threshold)
    
    return np.mean(f1_scores), np.std(f1_scores), np.mean(thresholds)

# Optuna 최적화 함수
def optimize_lgb(trial):
    # 기존 최적값 주변으로 범위 설정
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 50, 150),  # 기존: 89
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),  # 기존: 0.012
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.9),  # 기존: 0.70
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),  # 기존: 0.83
        'bagging_freq': 1,
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 80),  # 기존: 49
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 8.0),  # 기존: 2.67
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0),  # 기존: 5.40
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),  # 기존: 930
        'max_depth': trial.suggest_int('max_depth', 4, 8),  # 기존: 5
        'random_state': 42,
        'verbose': -1,
        'is_unbalance': True
    }
    
    f1_mean, _, _ = evaluate_lgb_cv(params, X_train, y_train)
    return f1_mean

print(f"\n=== Optuna 최적화 시작 (100 trials) ===")

# Optuna study 생성
study = optuna.create_study(
    direction='maximize', 
    sampler=optuna.samplers.TPESampler(seed=42)
)

# 최적화 실행
study.optimize(optimize_lgb, n_trials=100)

print(f"\n=== 최적화 완료 ===")
print(f"Best F1 Score: {study.best_value:.4f}")
print(f"Best Parameters:")
for key, value in study.best_params.items():
    print(f"- {key}: {value}")

# 최적 모델로 전체 CV 평가
best_params = {**study.best_params}
best_params.update({
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'bagging_freq': 1,
    'random_state': 42,
    'verbose': -1,
    'is_unbalance': True
})

f1_mean, f1_std, threshold_mean = evaluate_lgb_cv(best_params, X_train, y_train)

print(f"\n=== 최종 성능 ===")
print(f"CV F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")
print(f"Average Threshold: {threshold_mean:.3f}")

# 목표 달성 여부 확인
goal_achieved = f1_mean > 0.487
goal_status = "✅ 달성" if goal_achieved else "❌ 미달성"
print(f"목표 (F1 > 0.487): {goal_status}")

# 결과 저장
lgb_results = {
    'model_name': 'LightGBM',
    'cv_f1_mean': float(f1_mean),
    'cv_f1_std': float(f1_std),
    'threshold': float(threshold_mean),
    'best_params': best_params,
    'goal_achieved': bool(goal_achieved),
    'optuna_best_value': float(study.best_value),
    'n_trials': len(study.trials)
}

with open(RESULT_PATH + 'lgb_results.json', 'w') as f:
    json.dump(lgb_results, f, indent=2)

print(f"\n결과 저장: lgb_results.json")

# Trial 히스토리 저장 (상위 10개)
trials_df = study.trials_dataframe()
top_trials = trials_df.nlargest(10, 'value')[['number', 'value', 'params_num_leaves', 
                                              'params_learning_rate', 'params_n_estimators']]
top_trials.to_csv(RESULT_PATH + 'lgb_top_trials.csv', index=False)

print(f"상위 10 trials 저장: lgb_top_trials.csv")

print(f"\n=== Step 2 완료 ===")
print(f"다음 단계 실행: python step3_optimize_xgb.py") 