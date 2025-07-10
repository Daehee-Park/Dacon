import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
import warnings

warnings.filterwarnings('ignore')

# 데이터 경로 설정
RESULT_PATH = './result/'

print("=== Feature Selection Based Modeling ===")

# Target 로드
y_train = pd.read_csv(RESULT_PATH + 'y_train.csv')['Cancer']
test_ids = pd.read_csv(RESULT_PATH + 'test_ids.csv')['ID']

print(f"Target distribution: {y_train.value_counts().to_dict()}")
print(f"Positive rate: {y_train.mean():.4f}")

# 모든 feature selection 결과 로드
datasets = {}
dataset_names = ['original', 'rf_top30', 'lgb_top30', 'common_features', 
                'f_test_30', 'mutual_info_30', 'pca_20', 'pca_15']

for name in dataset_names:
    X_train = pd.read_csv(RESULT_PATH + f'X_train_{name}.csv')
    X_test = pd.read_csv(RESULT_PATH + f'X_test_{name}.csv')
    datasets[name] = (X_train, X_test)
    print(f"{name}: {X_train.shape}")

print("\n--- Model Evaluation on Different Feature Sets ---")

# Cross-validation 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 각 feature set에 대한 결과 저장
results = []

def evaluate_model(model, X_train, y_train, name, dataset_name):
    """모델 평가"""
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    mean_f1 = scores.mean()
    std_f1 = scores.std()
    
    # 추가로 AUC도 측정
    auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    mean_auc = auc_scores.mean()
    
    result = {
        'dataset': dataset_name,
        'model': name,
        'features': X_train.shape[1],
        'cv_f1_mean': mean_f1,
        'cv_f1_std': std_f1,
        'cv_auc_mean': mean_auc
    }
    
    print(f"{dataset_name:15} | {name:12} | Features: {X_train.shape[1]:2d} | F1: {mean_f1:.4f}±{std_f1:.4f} | AUC: {mean_auc:.4f}")
    
    return result

# 기본 모델들 정의
models = {
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        class_weight='balanced',
        verbose=-1
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1]
    ),
    'LogisticReg': LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    )
}

# 각 dataset과 model 조합 평가
for dataset_name, (X_train, X_test) in datasets.items():
    print(f"\n=== {dataset_name.upper()} Dataset ===")
    print("Dataset          | Model        | Features | CV F1 Score    | CV AUC")
    print("-" * 70)
    
    for model_name, model in models.items():
        result = evaluate_model(model, X_train, y_train, model_name, dataset_name)
        results.append(result)

# 결과 데이터프레임으로 정리
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('cv_f1_mean', ascending=False)

print(f"\n=== Top 10 Best Combinations ===")
print(results_df.head(10).to_string(index=False))

# 최고 성능 조합 선택
best_result = results_df.iloc[0]
best_dataset = best_result['dataset']
best_model_name = best_result['model']
best_f1 = best_result['cv_f1_mean']

print(f"\n=== Best Combination ===")
print(f"Dataset: {best_dataset}")
print(f"Model: {best_model_name}")
print(f"Features: {best_result['features']}")
print(f"CV F1: {best_f1:.4f}±{best_result['cv_f1_std']:.4f}")
print(f"CV AUC: {best_result['cv_auc_mean']:.4f}")

print(f"\n--- Hyperparameter Tuning for Best Combination ---")

# 최고 성능 데이터셋과 모델로 하이퍼파라미터 튜닝
X_train_best, X_test_best = datasets[best_dataset]

def objective(trial):
    """Optuna objective function"""
    if best_model_name == 'LightGBM':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'random_state': 42,
            'class_weight': 'balanced',
            'verbose': -1
        }
        model = lgb.LGBMClassifier(**params)
        
    elif best_model_name == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1
        }
        model = RandomForestClassifier(**params)
        
    elif best_model_name == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'scale_pos_weight': y_train.value_counts()[0]/y_train.value_counts()[1]
        }
        model = xgb.XGBClassifier(**params)
        
    else:  # LogisticRegression
        params = {
            'C': trial.suggest_float('C', 0.01, 100, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            'solver': 'liblinear',
            'random_state': 42,
            'class_weight': 'balanced',
            'max_iter': 1000
        }
        model = LogisticRegression(**params)
    
    # Cross-validation으로 F1 점수 계산
    scores = cross_val_score(model, X_train_best, y_train, cv=cv, scoring='f1', n_jobs=-1)
    return scores.mean()

# Optuna 최적화
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)

print(f"Best trial F1: {study.best_value:.4f}")
print(f"Best parameters: {study.best_params}")

# 최고 모델로 최종 학습 및 예측
print(f"\n--- Final Model Training & Prediction ---")

if best_model_name == 'LightGBM':
    final_model = lgb.LGBMClassifier(**study.best_params)
elif best_model_name == 'RandomForest':
    final_model = RandomForestClassifier(**study.best_params)
elif best_model_name == 'XGBoost':
    final_model = xgb.XGBClassifier(**study.best_params)
else:
    final_model = LogisticRegression(**study.best_params)

# 최종 모델 학습
final_model.fit(X_train_best, y_train)

# Train set에서 성능 확인
y_pred_train = final_model.predict(X_train_best)
y_pred_proba_train = final_model.predict_proba(X_train_best)[:, 1]

train_f1 = f1_score(y_train, y_pred_train)
train_auc = roc_auc_score(y_train, y_pred_proba_train)

print(f"Final Model Performance on Train:")
print(f"F1 Score: {train_f1:.4f}")
print(f"AUC Score: {train_auc:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_train, y_pred_train))

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_train, y_pred_train))

# Test set 예측
y_pred_test_proba = final_model.predict_proba(X_test_best)[:, 1]

# 임계값 최적화 (Train set에서)
thresholds = np.arange(0.1, 0.9, 0.01)
best_threshold = 0.5
best_f1_thresh = 0

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba_train >= threshold).astype(int)
    f1_thresh = f1_score(y_train, y_pred_thresh)
    if f1_thresh > best_f1_thresh:
        best_f1_thresh = f1_thresh
        best_threshold = threshold

print(f"\nThreshold Optimization:")
print(f"Best threshold: {best_threshold:.3f}")
print(f"F1 with best threshold: {best_f1_thresh:.4f}")

# 최적 임계값으로 최종 예측
y_pred_test = (y_pred_test_proba >= best_threshold).astype(int)

print(f"\nTest Predictions:")
print(f"Predicted positive rate: {y_pred_test.mean():.4f}")
print(f"Prediction distribution: {pd.Series(y_pred_test).value_counts().to_dict()}")

# 제출 파일 생성
submission_df = pd.DataFrame({
    'ID': test_ids,
    'Cancer': y_pred_test
})

submission_df.to_csv(RESULT_PATH + 'submission_feature_selection.csv', index=False)

# 결과 요약 저장
summary = {
    'best_dataset': best_dataset,
    'best_model': best_model_name,
    'best_features': int(best_result['features']),
    'cv_f1_mean': float(best_result['cv_f1_mean']),
    'cv_f1_std': float(best_result['cv_f1_std']),
    'cv_auc_mean': float(best_result['cv_auc_mean']),
    'tuned_f1': float(study.best_value),
    'final_train_f1': float(train_f1),
    'final_train_auc': float(train_auc),
    'best_threshold': float(best_threshold),
    'threshold_f1': float(best_f1_thresh),
    'best_params': study.best_params
}

import json
with open(RESULT_PATH + 'model_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# 전체 결과 저장
results_df.to_csv(RESULT_PATH + 'all_combinations_results.csv', index=False)

print(f"\n=== Feature Selection Modeling Complete ===")
print(f"Best combination: {best_dataset} + {best_model_name}")
print(f"Features reduced from original to {best_result['features']}")
print(f"Final F1 Score: {train_f1:.4f}")
print(f"Submission saved: submission_feature_selection.csv")
print(f"Model summary saved: model_summary.json") 