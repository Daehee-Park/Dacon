import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
import warnings

# 분류 모델들 import (SVM 제외)
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier
)
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, 
    PassiveAggressiveClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Gradient Boosting 라이브러리들
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

# 데이터 경로 설정
RESULT_PATH = './result/'

print("=== Fast Model Comparison (No SVM) ===")

# 데이터 로드 함수
def load_data(data_type='optimized'):
    if data_type == 'optimized':
        X_train = pd.read_csv(RESULT_PATH + 'X_train_optimized.csv')
        X_test = pd.read_csv(RESULT_PATH + 'X_test_optimized.csv')
    else:  # original
        X_train = pd.read_csv(RESULT_PATH + 'X_train_original.csv')
        X_test = pd.read_csv(RESULT_PATH + 'X_test_original.csv')
    
    y_train = pd.read_csv(RESULT_PATH + 'y_train.csv')['Cancer']
    test_ids = pd.read_csv(RESULT_PATH + 'test_ids.csv')['ID']
    
    return X_train, X_test, y_train, test_ids

# Cross-validation 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n--- Defining Fast Classification Models ---")

def get_models(y_train):
    """모델 정의 함수"""
    return {
        # Gradient Boosting Libraries (핵심)
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        ),
        
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=y_train.value_counts()[0]/y_train.value_counts()[1]
        ),
        
        'CatBoost': cb.CatBoostClassifier(
            n_estimators=100,
            random_state=42,
            class_weights=[1, y_train.value_counts()[0]/y_train.value_counts()[1]],
            verbose=False
        ),
        
        # Tree-based Ensemble Models
        'RandomForest': RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced',
            n_jobs=-1
        ),
        
        'ExtraTrees': ExtraTreesClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced',
            n_jobs=-1
        ),
        
        # Linear Models (빠름)
        'LogisticRegression': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        ),
        
        'RidgeClassifier': RidgeClassifier(
            random_state=42,
            class_weight='balanced'
        ),
        
        # Naive Bayes (매우 빠름)
        'GaussianNB': GaussianNB(),
        'BernoulliNB': BernoulliNB(),
        
        # Tree Models
        'DecisionTree': DecisionTreeClassifier(
            random_state=42,
            class_weight='balanced'
        ),
        
        # Neural Network (중간 속도)
        'MLPClassifier': MLPClassifier(
            hidden_layer_sizes=(100,),
            random_state=42,
            max_iter=300
        ),
        
        # 추가 앙상블
        'Bagging': BaggingClassifier(
            n_estimators=10,
            random_state=42
        )
    }

def evaluate_models(X_train, y_train, data_name):
    """모델 평가 함수"""
    print(f"\n=== {data_name.upper()} DATA EVALUATION ===")
    print(f"Data shape: {X_train.shape}")
    print("Model                | CV F1 Score    | Training Time | Notes")
    print("-" * 70)
    
    models = get_models(y_train)
    results = []
    
    for model_name, model in models.items():
        try:
            start_time = time.time()
            
            # Cross-validation으로 F1 점수 계산
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
            
            training_time = time.time() - start_time
            mean_f1 = cv_scores.mean()
            std_f1 = cv_scores.std()
            
            result = {
                'data_type': data_name,
                'model': model_name,
                'cv_f1_mean': mean_f1,
                'cv_f1_std': std_f1,
                'training_time': training_time,
                'cv_scores': cv_scores.tolist(),
                'status': 'success'
            }
            
            results.append(result)
            
            print(f"{model_name:20} | {mean_f1:.4f}±{std_f1:.4f} | {training_time:8.2f}s | Success")
            
        except Exception as e:
            result = {
                'data_type': data_name,
                'model': model_name,
                'cv_f1_mean': 0.0,
                'cv_f1_std': 0.0,
                'training_time': 0.0,
                'cv_scores': [],
                'status': f'failed: {str(e)[:50]}'
            }
            
            results.append(result)
            print(f"{model_name:20} | Failed        | Failed    | {str(e)[:30]}")
    
    return results

# 1. Original Data 평가
print("Loading original data...")
X_train_orig, X_test_orig, y_train, test_ids = load_data('original')
results_original = evaluate_models(X_train_orig, y_train, 'original')

# 2. Optimized Data 평가
print("\nLoading optimized data...")
X_train_opt, X_test_opt, y_train, test_ids = load_data('optimized')
results_optimized = evaluate_models(X_train_opt, y_train, 'optimized')

# 결과 통합
all_results = results_original + results_optimized
results_df = pd.DataFrame(all_results)

# 성공한 결과만 필터링
successful_results = results_df[results_df['status'] == 'success'].copy()

print(f"\n=== COMPARISON: Original vs Optimized ===")
print("Data Type | Model                | CV F1 Score    | Features | Time")
print("-" * 75)

# 모델별로 원본 vs 최적화 비교
models_list = successful_results['model'].unique()
for model in models_list:
    orig_result = successful_results[(successful_results['model'] == model) & 
                                   (successful_results['data_type'] == 'original')]
    opt_result = successful_results[(successful_results['model'] == model) & 
                                  (successful_results['data_type'] == 'optimized')]
    
    if len(orig_result) > 0 and len(opt_result) > 0:
        orig_f1 = orig_result.iloc[0]['cv_f1_mean']
        opt_f1 = opt_result.iloc[0]['cv_f1_mean']
        orig_time = orig_result.iloc[0]['training_time']
        opt_time = opt_result.iloc[0]['training_time']
        
        print(f"Original  | {model:20} | {orig_f1:.4f}±{orig_result.iloc[0]['cv_f1_std']:.4f} | {X_train_orig.shape[1]:8d} | {orig_time:6.2f}s")
        print(f"Optimized | {model:20} | {opt_f1:.4f}±{opt_result.iloc[0]['cv_f1_std']:.4f} | {X_train_opt.shape[1]:8d} | {opt_time:6.2f}s")
        
        # 성능 비교
        improvement = ((opt_f1 - orig_f1) / orig_f1) * 100 if orig_f1 > 0 else 0
        if improvement > 1:
            print(f"          | {'':20} | ↗️ +{improvement:.1f}% improvement")
        elif improvement < -1:
            print(f"          | {'':20} | ↘️ {improvement:.1f}% decline")
        else:
            print(f"          | {'':20} | ≈ Similar performance")
        print("-" * 75)

# Top 5 모델 (전체 중에서)
print(f"\n=== TOP 5 MODELS (All Data Types) ===")
top_performers = successful_results.nlargest(5, 'cv_f1_mean')

for i, (_, row) in enumerate(top_performers.iterrows(), 1):
    print(f"{i}. {row['model']} ({row['data_type']}): F1 = {row['cv_f1_mean']:.4f}±{row['cv_f1_std']:.4f}")

# LightGBM 상세 분석
print(f"\n=== LIGHTGBM DETAILED ANALYSIS ===")
lgb_results = successful_results[successful_results['model'] == 'LightGBM']
for _, row in lgb_results.iterrows():
    print(f"{row['data_type'].capitalize()} Data:")
    print(f"  F1 Score: {row['cv_f1_mean']:.4f}±{row['cv_f1_std']:.4f}")
    print(f"  CV Scores: {[f'{score:.4f}' for score in row['cv_scores']]}")
    print(f"  Training Time: {row['training_time']:.2f}s")

# 결과 저장
results_df.to_csv(RESULT_PATH + 'fast_model_comparison.csv', index=False)
top_performers.to_csv(RESULT_PATH + 'top5_fast_models.csv', index=False)

# 요약 통계
summary = {
    'original_features': int(X_train_orig.shape[1]),
    'optimized_features': int(X_train_opt.shape[1]),
    'feature_reduction': f"{((X_train_orig.shape[1] - X_train_opt.shape[1]) / X_train_orig.shape[1] * 100):.1f}%",
    'best_model_overall': top_performers.iloc[0]['model'],
    'best_f1_overall': float(top_performers.iloc[0]['cv_f1_mean']),
    'best_data_type': top_performers.iloc[0]['data_type'],
    'models_evaluated': len(models_list),
    'successful_evaluations': len(successful_results)
}

import json
with open(RESULT_PATH + 'fast_comparison_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n=== SUMMARY ===")
print(f"Original features: {X_train_orig.shape[1]}")
print(f"Optimized features: {X_train_opt.shape[1]} ({summary['feature_reduction']} reduction)")
print(f"Best overall: {summary['best_model_overall']} on {summary['best_data_type']} data")
print(f"Best F1 score: {summary['best_f1_overall']:.4f}")

print(f"\nFiles saved:")
print(f"- fast_model_comparison.csv")
print(f"- top5_fast_models.csv") 
print(f"- fast_comparison_summary.json")

print(f"\n빠른 모델 비교 완료!") 