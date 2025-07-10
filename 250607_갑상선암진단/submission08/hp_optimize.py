import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
import warnings

warnings.filterwarnings('ignore')

# 데이터 경로 설정
RESULT_PATH = './result/'

print("=== Top 3 Models Hyperparameter Optimization ===")

# 데이터 로드 (Original data 사용)
X_train = pd.read_csv(RESULT_PATH + 'X_train_original.csv')
X_test = pd.read_csv(RESULT_PATH + 'X_test_original.csv')
y_train = pd.read_csv(RESULT_PATH + 'y_train.csv')['Cancer']
test_ids = pd.read_csv(RESULT_PATH + 'test_ids.csv')['ID']

print(f"Data shape: {X_train.shape}")
print(f"Target distribution: {y_train.value_counts().to_dict()}")
print(f"Positive rate: {y_train.mean():.4f}")

# Cross-validation 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 기본 성능 저장
baseline_results = {
    'LightGBM': 0.4815,
    'XGBoost': 0.4238,
    'CatBoost': 0.4095
}

print(f"\nBaseline performances:")
for model, score in baseline_results.items():
    print(f"- {model}: {score:.4f}")

print(f"\n--- Hyperparameter Optimization ---")

# 1. LightGBM 최적화
def optimize_lightgbm():
    print(f"\n1. LightGBM Optimization")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'random_state': 42,
            'class_weight': 'balanced',
            'verbose': -1
        }
        
        model = lgb.LGBMClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    return study.best_params, study.best_value

# 2. XGBoost 최적화
def optimize_xgboost():
    print(f"\n2. XGBoost Optimization")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'scale_pos_weight': y_train.value_counts()[0]/y_train.value_counts()[1]
        }
        
        model = xgb.XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    return study.best_params, study.best_value

# 3. CatBoost 최적화
def optimize_catboost():
    print(f"\n3. CatBoost Optimization")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'random_state': 42,
            'auto_class_weights': 'Balanced',
            'verbose': False
        }
        
        model = cb.CatBoostClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    return study.best_params, study.best_value

# 최적화 실행
optimization_results = {}

print("Starting hyperparameter optimization...")
start_time = time.time()

# LightGBM 최적화
lgb_params, lgb_score = optimize_lightgbm()
optimization_results['LightGBM'] = {
    'best_params': lgb_params,
    'best_score': lgb_score,
    'baseline_score': baseline_results['LightGBM'],
    'improvement': ((lgb_score - baseline_results['LightGBM']) / baseline_results['LightGBM']) * 100
}

print(f"LightGBM - Best F1: {lgb_score:.4f} (Baseline: {baseline_results['LightGBM']:.4f})")
print(f"Improvement: {optimization_results['LightGBM']['improvement']:+.2f}%")

# XGBoost 최적화
xgb_params, xgb_score = optimize_xgboost()
optimization_results['XGBoost'] = {
    'best_params': xgb_params,
    'best_score': xgb_score,
    'baseline_score': baseline_results['XGBoost'],
    'improvement': ((xgb_score - baseline_results['XGBoost']) / baseline_results['XGBoost']) * 100
}

print(f"XGBoost - Best F1: {xgb_score:.4f} (Baseline: {baseline_results['XGBoost']:.4f})")
print(f"Improvement: {optimization_results['XGBoost']['improvement']:+.2f}%")

# CatBoost 최적화
cat_params, cat_score = optimize_catboost()
optimization_results['CatBoost'] = {
    'best_params': cat_params,
    'best_score': cat_score,
    'baseline_score': baseline_results['CatBoost'],
    'improvement': ((cat_score - baseline_results['CatBoost']) / baseline_results['CatBoost']) * 100
}

print(f"CatBoost - Best F1: {cat_score:.4f} (Baseline: {baseline_results['CatBoost']:.4f})")
print(f"Improvement: {optimization_results['CatBoost']['improvement']:+.2f}%")

total_time = time.time() - start_time

print(f"\n=== Optimization Results Summary ===")
print(f"Total optimization time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")

# 결과 정렬 (성능순)
sorted_results = sorted(optimization_results.items(), key=lambda x: x[1]['best_score'], reverse=True)

print(f"\nRanking after optimization:")
for i, (model, result) in enumerate(sorted_results, 1):
    print(f"{i}. {model}: {result['best_score']:.4f} ({result['improvement']:+.2f}%)")

# 최고 모델로 최종 학습 및 예측
print(f"\n--- Final Model Training & Prediction ---")
best_model_name = sorted_results[0][0]
best_params = sorted_results[0][1]['best_params']
best_score = sorted_results[0][1]['best_score']

print(f"Best model: {best_model_name}")
print(f"Best CV F1: {best_score:.4f}")

# 최고 모델 학습
if best_model_name == 'LightGBM':
    final_model = lgb.LGBMClassifier(**best_params)
elif best_model_name == 'XGBoost':
    final_model = xgb.XGBClassifier(**best_params)
else:  # CatBoost
    final_model = cb.CatBoostClassifier(**best_params)

# 최종 모델 학습
final_model.fit(X_train, y_train)

# Train set 성능 확인
y_pred_train = final_model.predict(X_train)
y_pred_proba_train = final_model.predict_proba(X_train)[:, 1]

train_f1 = f1_score(y_train, y_pred_train)
train_auc = roc_auc_score(y_train, y_pred_proba_train)

print(f"\nFinal model performance on train:")
print(f"F1 Score: {train_f1:.4f}")
print(f"AUC Score: {train_auc:.4f}")

# 임계값 최적화
thresholds = np.arange(0.1, 0.9, 0.01)
best_threshold = 0.5
best_f1_thresh = 0

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba_train >= threshold).astype(int)
    f1_thresh = f1_score(y_train, y_pred_thresh)
    if f1_thresh > best_f1_thresh:
        best_f1_thresh = f1_thresh
        best_threshold = threshold

print(f"\nThreshold optimization:")
print(f"Best threshold: {best_threshold:.3f}")
print(f"F1 with best threshold: {best_f1_thresh:.4f}")

# Test set 예측
y_pred_test_proba = final_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_pred_test_proba >= best_threshold).astype(int)

print(f"\nTest predictions:")
print(f"Predicted positive rate: {y_pred_test.mean():.4f}")
print(f"Prediction distribution: {pd.Series(y_pred_test).value_counts().to_dict()}")

# 제출 파일 생성
submission_df = pd.DataFrame({
    'ID': test_ids,
    'Cancer': y_pred_test
})

submission_df.to_csv(RESULT_PATH + 'submission_hp_optimized.csv', index=False)

# 결과 저장
import json

# 최적화 결과 저장
with open(RESULT_PATH + 'hp_optimization_results.json', 'w') as f:
    json.dump(optimization_results, f, indent=2)

# 최종 모델 정보 저장
final_summary = {
    'best_model': best_model_name,
    'best_cv_f1': float(best_score),
    'final_train_f1': float(train_f1),
    'final_train_auc': float(train_auc),
    'best_threshold': float(best_threshold),
    'threshold_f1': float(best_f1_thresh),
    'predicted_positive_rate': float(y_pred_test.mean()),
    'optimization_time_minutes': total_time / 60,
    'best_params': best_params
}

with open(RESULT_PATH + 'final_model_summary.json', 'w') as f:
    json.dump(final_summary, f, indent=2)

print(f"\n=== Final Summary ===")
print(f"Best model: {best_model_name}")
print(f"CV F1 Score: {best_score:.4f}")
print(f"Final F1 Score: {train_f1:.4f}")
print(f"Optimized threshold: {best_threshold:.3f}")
print(f"Threshold F1 Score: {best_f1_thresh:.4f}")

print(f"\nFiles saved:")
print(f"- submission_hp_optimized.csv: 제출 파일")
print(f"- hp_optimization_results.json: 최적화 결과")
print(f"- final_model_summary.json: 최종 모델 요약")

print(f"\n하이퍼파라미터 최적화 완료!") 