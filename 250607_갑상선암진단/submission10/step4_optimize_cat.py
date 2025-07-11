import pandas as pd
import numpy as np
import catboost as cb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import warnings
import json
import os

warnings.filterwarnings('ignore')

RESULT_PATH = './result/'

print("=== Step 4: CatBoost Optimization ===")

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
reference_params = {
    'n_estimators': 945,
    'learning_rate': 0.0161,
    'depth': 5,
    'l2_leaf_reg': 4.297,
    'random_strength': 4.561
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

def evaluate_cat_cv(params, X, y, cv_folds=5):
    """CatBoost CV 평가"""
    f1_scores = []
    thresholds = []
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 모델 학습
        model = cb.CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        
        # 예측 확률
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # 최적 threshold 및 F1 score
        threshold, f1 = optimize_threshold(y_val, y_prob)
        f1_scores.append(f1)
        thresholds.append(threshold)
    
    return np.mean(f1_scores), np.std(f1_scores), np.mean(thresholds)

# Optuna 최적화 함수
def optimize_cat(trial):
    # 기존 최적값 주변으로 범위 설정
    params = {
        'objective': 'Logloss',
        'iterations': trial.suggest_int('iterations', 500, 1500),  # 기존: 945
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),  # 기존: 0.016
        'depth': trial.suggest_int('depth', 4, 8),  # 기존: 5
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 8.0),  # 기존: 4.30
        'random_strength': trial.suggest_float('random_strength', 1.0, 8.0),  # 기존: 4.56
        'border_count': trial.suggest_int('border_count', 50, 255),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_state': 42,
        'verbose': False,
        'auto_class_weights': 'Balanced',
        'early_stopping_rounds': 50
    }
    
    f1_mean, _, _ = evaluate_cat_cv(params, X_train, y_train)
    return f1_mean

print(f"\n=== Optuna 최적화 시작 (100 trials) ===")

# Optuna study 생성
study = optuna.create_study(
    direction='maximize', 
    sampler=optuna.samplers.TPESampler(seed=42)
)

# 최적화 실행
study.optimize(optimize_cat, n_trials=100)

print(f"\n=== 최적화 완료 ===")
print(f"Best F1 Score: {study.best_value:.4f}")
print(f"Best Parameters:")
for key, value in study.best_params.items():
    print(f"- {key}: {value}")

# 최적 모델로 전체 CV 평가
best_params = {**study.best_params}
best_params.update({
    'objective': 'Logloss',
    'random_state': 42,
    'verbose': False,
    'auto_class_weights': 'Balanced'
})

# early_stopping_rounds 제거 (CV 평가시)
if 'early_stopping_rounds' in best_params:
    del best_params['early_stopping_rounds']

f1_mean, f1_std, threshold_mean = evaluate_cat_cv(best_params, X_train, y_train)

print(f"\n=== 최종 성능 ===")
print(f"CV F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")
print(f"Average Threshold: {threshold_mean:.3f}")

# 목표 달성 여부 확인
goal_achieved = f1_mean > 0.487
goal_status = "✅ 달성" if goal_achieved else "❌ 미달성"
print(f"목표 (F1 > 0.487): {goal_status}")

# 결과 저장
cat_results = {
    'model_name': 'CatBoost',
    'cv_f1_mean': float(f1_mean),
    'cv_f1_std': float(f1_std),
    'threshold': float(threshold_mean),
    'best_params': best_params,
    'goal_achieved': goal_achieved,
    'optuna_best_value': float(study.best_value),
    'n_trials': len(study.trials)
}

with open(RESULT_PATH + 'cat_results.json', 'w') as f:
    json.dump(cat_results, f, indent=2)

print(f"\n결과 저장: cat_results.json")

# Trial 히스토리 저장 (상위 10개)
trials_df = study.trials_dataframe()
top_trials = trials_df.nlargest(10, 'value')[['number', 'value', 'params_depth', 
                                              'params_learning_rate', 'params_iterations']]
top_trials.to_csv(RESULT_PATH + 'cat_top_trials.csv', index=False)

print(f"상위 10 trials 저장: cat_top_trials.csv")

print(f"\n=== Step 4 완료 ===")
print(f"다음 단계 실행: python step5_optimize_et.py") 