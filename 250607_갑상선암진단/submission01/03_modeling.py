import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import optuna
import warnings

warnings.filterwarnings('ignore')

# 데이터 경로 설정
DATA_PATH = './data/'

print("--- 1. Loading Processed Data ---")
# 전처리된 데이터 불러오기
X = pd.read_csv(DATA_PATH + 'X_train_processed.csv')
y = pd.read_csv(DATA_PATH + 'y_train_processed.csv').squeeze()
X_test = pd.read_csv(DATA_PATH + 'X_test_processed.csv')

print(f"Train shape: {X.shape}, Test shape: {X_test.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# 클래스 비율 확인
pos_ratio = y.sum() / len(y)
print(f"Positive ratio: {pos_ratio:.4f}")

print("\n--- 2. Recall 최적화 전략 ---")

def recall_focused_objective(trial):
    """Recall을 우선시하는 objective function"""
    
    # 더 aggressive한 파라미터 범위
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'num_leaves': trial.suggest_int('num_leaves', 10, 80),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 12.0),  # 클래스 가중치 조정
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # 3-fold CV로 빠르게
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        
        # Recall을 높이는 낮은 threshold 시도
        val_proba = model.predict_proba(X_val)[:, 1]
        
        # Threshold 최적화 - F1과 Recall의 가중합
        best_score = 0
        for threshold in np.arange(0.05, 0.6, 0.05):  # 더 낮은 threshold
            val_preds = (val_proba >= threshold).astype(int)
            f1 = f1_score(y_val, val_preds)
            recall = (val_preds & y_val).sum() / y_val.sum() if y_val.sum() > 0 else 0
            
            # F1과 Recall의 가중 평균 (Recall에 더 큰 가중치)
            combined_score = 0.3 * f1 + 0.7 * recall
            best_score = max(best_score, combined_score)
        
        f1_scores.append(best_score)
    
    return np.mean(f1_scores)

# Optuna study
study = optuna.create_study(direction='maximize', study_name='recall_focused')
study.optimize(recall_focused_objective, n_trials=30, show_progress_bar=True)

print(f"Best Combined Score: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

print("\n--- 3. Multi-Model Ensemble ---")

# 최적 파라미터
best_params = study.best_params.copy()
best_params.update({
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1
})

# 여러 모델 설정
models_config = {
    'lgb_recall': best_params,
    'lgb_balanced': {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': 500,
        'learning_rate': 0.05,
        'num_leaves': 50,
        'max_depth': 6,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.01,
        'reg_lambda': 0.01,
        'is_unbalance': True,  # scale_pos_weight 대신 is_unbalance 사용
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1
    },
    'rf_balanced': {
        'n_estimators': 300,
        'max_depth': 8,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
}

# Cross-validation
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

model_results = {}
oof_predictions = {}
test_predictions = {}

for model_name, config in models_config.items():
    print(f"\n--- Training {model_name} ---")
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 모델별 학습
        if model_name.startswith('lgb'):
            model = lgb.LGBMClassifier(**config)
            model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        else:  # RandomForest
            model = RandomForestClassifier(**config)
            model.fit(X_train, y_train)
        
        # 예측 및 최적 threshold 찾기
        val_proba = model.predict_proba(X_val)[:, 1]
        
        best_f1 = 0
        best_threshold = 0.5
        for threshold in np.arange(0.1, 0.8, 0.02):
            val_preds = (val_proba >= threshold).astype(int)
            f1 = f1_score(y_val, val_preds)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        fold_scores.append({
            'f1': best_f1,
            'threshold': best_threshold,
            'auc': roc_auc_score(y_val, val_proba)
        })
        
        oof_preds[val_idx] = val_proba
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    
    # 결과 저장
    model_results[model_name] = fold_scores
    oof_predictions[model_name] = oof_preds
    test_predictions[model_name] = test_preds
    
    # 평균 성능 출력
    avg_f1 = np.mean([s['f1'] for s in fold_scores])
    avg_auc = np.mean([s['auc'] for s in fold_scores])
    print(f"{model_name}: F1={avg_f1:.4f}, AUC={avg_auc:.4f}")

print("\n--- 4. Weighted Ensemble ---")

# 성능 기반 가중치 계산
model_weights = {}
for model_name, scores in model_results.items():
    avg_f1 = np.mean([s['f1'] for s in scores])
    model_weights[model_name] = avg_f1

# 정규화
total_weight = sum(model_weights.values())
model_weights = {k: v/total_weight for k, v in model_weights.items()}

print("Model weights:")
for model_name, weight in model_weights.items():
    print(f"  {model_name}: {weight:.3f}")

# 앙상블 예측
ensemble_oof = np.zeros(len(X))
ensemble_test = np.zeros(len(X_test))

for model_name, weight in model_weights.items():
    ensemble_oof += oof_predictions[model_name] * weight
    ensemble_test += test_predictions[model_name] * weight

# 앙상블 최적 threshold
best_f1 = 0
best_threshold = 0.5
best_preds = None

for threshold in np.arange(0.1, 0.8, 0.01):
    preds = (ensemble_oof >= threshold).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_preds = preds

ensemble_auc = roc_auc_score(y, ensemble_oof)

print(f"\nFinal Ensemble:")
print(f"F1: {best_f1:.4f}, Threshold: {best_threshold:.3f}, AUC: {ensemble_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y, best_preds))

print("\n--- 5. Saving Results ---")
np.save(DATA_PATH + 'ensemble_test_preds.npy', ensemble_test)
np.save(DATA_PATH + 'ensemble_oof_preds.npy', ensemble_oof)

import pickle
with open(DATA_PATH + 'ensemble_threshold.pkl', 'wb') as f:
    pickle.dump(best_threshold, f)

print("Ensemble predictions saved successfully.")