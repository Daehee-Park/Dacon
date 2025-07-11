#!/usr/bin/env python3
"""
10차 시도: 하이퍼파라미터 최적화 + 딥러닝 전략
- 필수 전처리만
- 7개 모델: LGBM, XGBoost, CatBoost, ExtraTrees, Logistic, MLP, TabNet
- CV Score 비교 후 최고 모델 선택
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, make_scorer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import optuna
import warnings
warnings.filterwarnings('ignore')

# PyTorch 관련
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin

# TabNet 관련 (설치 필요)
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("Warning: pytorch-tabnet not installed. TabNet will be skipped.")

print("=== 10차 시도: 하이퍼파라미터 최적화 + 딥러닝 전략 ===")

# 1. 데이터 로드
print("\n1. 데이터 로드")
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Class distribution: {train['Cancer'].value_counts().to_dict()}")

# 2. 필수 전처리만 진행
print("\n2. 필수 전처리")

def minimal_preprocessing(train_df, test_df):
    """최소한의 전처리만 수행"""
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    # ID 저장 후 제거
    test_ids = test_processed['ID'].copy()
    train_processed = train_processed.drop('ID', axis=1)
    test_processed = test_processed.drop('ID', axis=1)
    
    # Target 분리
    y = train_processed['Cancer'].copy()
    X = train_processed.drop('Cancer', axis=1)
    
    # 결측치 처리 (중앙값/최빈값)
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if X[col].dtype in ['int64', 'float64']:
                fill_val = X[col].median()
                X[col].fillna(fill_val, inplace=True)
                test_processed[col].fillna(fill_val, inplace=True)
            else:
                fill_val = X[col].mode()[0]
                X[col].fillna(fill_val, inplace=True)
                test_processed[col].fillna(fill_val, inplace=True)
    
    # 범주형 변수 인코딩
    categorical_cols = X.select_dtypes(include=['object']).columns
    le_dict = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        combined_values = pd.concat([X[col], test_processed[col]])
        le.fit(combined_values)
        X[col] = le.transform(X[col])
        test_processed[col] = le.transform(test_processed[col])
        le_dict[col] = le
    
    # 수치형 변수 스케일링
    scaler = StandardScaler()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    test_processed[numeric_cols] = scaler.transform(test_processed[numeric_cols])
    
    return X, y, test_processed, test_ids

X, y, X_test, test_ids = minimal_preprocessing(train, test)
print(f"전처리 완료: X shape {X.shape}, y shape {y.shape}")
print(f"특성 수: {X.shape[1]}개")

# 3. PyTorch MLP 모델 정의
class MLPClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layers=[128, 64], dropout=0.3, lr=0.001, 
                 batch_size=64, epochs=100, optimizer='adam'):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _build_model(self, input_dim):
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        
        # 데이터를 텐서로 변환
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y).reshape(-1, 1)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # 모델 생성
        self.model = self._build_model(self.n_features_).to(self.device)
        
        # 옵티마이저 선택
        if self.optimizer == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        
        criterion = nn.BCELoss()
        
        # 학습
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        return self
    
    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(self.device)
        
        with torch.no_grad():
            proba = self.model(X_tensor).cpu().numpy()
        
        # sklearn 형식으로 변환 (negative, positive)
        return np.column_stack([1 - proba.flatten(), proba.flatten()])
    
    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)

# 4. F1 최적화 함수
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

def evaluate_model_cv(model, X, y, cv_folds=5):
    """CV로 모델 평가"""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    f1_scores = []
    thresholds = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 모델 학습
        model.fit(X_train, y_train)
        
        # 예측 확률
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_val)[:, 1]
        else:
            y_prob = model.predict(X_val)
        
        # 최적 threshold 및 F1 score
        threshold, f1 = optimize_threshold(y_val, y_prob)
        f1_scores.append(f1)
        thresholds.append(threshold)
    
    return np.mean(f1_scores), np.std(f1_scores), np.mean(thresholds)

# 5. 모델별 하이퍼파라미터 최적화
print("\n3. 하이퍼파라미터 최적화 시작")

results = {}

# 5.1 LightGBM 최적화
print("\n5.1 LightGBM 최적화")
def optimize_lgb(trial):
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': 1,
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'random_state': 42,
        'verbose': -1,
        'is_unbalance': True
    }
    
    model = lgb.LGBMClassifier(**params)
    f1_mean, _, _ = evaluate_model_cv(model, X, y)
    return f1_mean

study_lgb = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study_lgb.optimize(optimize_lgb, n_trials=100)

best_lgb = lgb.LGBMClassifier(**study_lgb.best_params, random_state=42, verbose=-1, is_unbalance=True)
lgb_f1, lgb_std, lgb_threshold = evaluate_model_cv(best_lgb, X, y)
results['LightGBM'] = {'f1': lgb_f1, 'std': lgb_std, 'threshold': lgb_threshold, 'params': study_lgb.best_params}
print(f"LightGBM - F1: {lgb_f1:.4f} ± {lgb_std:.4f}, Threshold: {lgb_threshold:.3f}")

print("=== 진행 완료: LightGBM ===")

# 임시로 LightGBM만 테스트
print("\n=== 모델 성능 비교 ===")
sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)

for i, (model_name, result) in enumerate(sorted_results, 1):
    print(f"{i}. {model_name}: F1 = {result['f1']:.4f} ± {result['std']:.4f} (Threshold: {result['threshold']:.3f})")

# 최고 성능 모델로 최종 예측
best_model_name, best_result = sorted_results[0]
print(f"\n최고 성능 모델: {best_model_name} (F1: {best_result['f1']:.4f})")

# 7. 최종 예측 및 제출 파일 생성
print("\n7. 최종 예측")

# LightGBM으로 전체 데이터 학습
final_model = lgb.LGBMClassifier(**best_result['params'], random_state=42, verbose=-1, is_unbalance=True)

# 전체 데이터로 학습
final_model.fit(X, y)

# 테스트 데이터 예측
test_proba = final_model.predict_proba(X_test)[:, 1]

# 최적 threshold 적용
final_threshold = best_result['threshold']
test_pred = (test_proba >= final_threshold).astype(int)

# 제출 파일 생성
submission = pd.DataFrame({
    'ID': test_ids,
    'Cancer': test_pred
})

submission.to_csv('submission.csv', index=False)
print(f"제출 파일 생성 완료: submission.csv")
print(f"예측 양성률: {test_pred.mean():.1%} ({test_pred.sum()}/{len(test_pred)})")

# 8. 결과 요약 저장
print("\n8. 결과 요약")
summary = {
    'best_model': best_model_name,
    'best_f1': best_result['f1'],
    'best_threshold': best_result['threshold'],
    'all_results': results
}

import json
with open('results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print("=== 10차 시도 (LightGBM 테스트) 완료 ===")
print(f"최고 성능: {best_model_name} - F1: {best_result['f1']:.4f}")
print(f"목표 달성: {'✅' if best_result['f1'] > 0.49 else '❌'} (목표 F1 > 0.49)")

print("\n다음 단계: 다른 모델들 추가 최적화 필요")
print("- XGBoost, CatBoost, ExtraTrees, LogisticRegression")
print("- MLP(PyTorch), TabNet 딥러닝 모델") 