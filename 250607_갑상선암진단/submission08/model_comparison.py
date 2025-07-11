import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings

# 분류 모델들 import
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, BaggingClassifier, VotingClassifier
)
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier, 
    PassiveAggressiveClassifier, Perceptron
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
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

print("=== Comprehensive Model Comparison with Default Parameters ===")

# 데이터 로드
X_train = pd.read_csv(RESULT_PATH + 'X_train_optimized.csv')
X_test = pd.read_csv(RESULT_PATH + 'X_test_optimized.csv')
y_train = pd.read_csv(RESULT_PATH + 'y_train.csv')['Cancer']
test_ids = pd.read_csv(RESULT_PATH + 'test_ids.csv')['ID']

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Target distribution: {y_train.value_counts().to_dict()}")

# Cross-validation 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"\n--- Defining Classification Models (Default Parameters) ---")

# 모델 정의 (기본 파라미터 + 클래스 불균형 고려)
models = {
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
    
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100,
        random_state=42
    ),
    
    'AdaBoost': AdaBoostClassifier(
        n_estimators=50,  # 기본값이 50
        random_state=42,
        algorithm='SAMME'  # 안정성을 위해
    ),
    
    # Gradient Boosting Libraries
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
    
    # Linear Models
    'LogisticRegression': LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    ),
    
    'RidgeClassifier': RidgeClassifier(
        random_state=42,
        class_weight='balanced'
    ),
    
    'SGDClassifier': SGDClassifier(
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    ),
    
    'PassiveAggressive': PassiveAggressiveClassifier(
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    ),
    
    # Tree Models
    'DecisionTree': DecisionTreeClassifier(
        random_state=42,
        class_weight='balanced'
    ),
    
    # Naive Bayes
    'GaussianNB': GaussianNB(),
    'BernoulliNB': BernoulliNB(),
    
    # SVM Models
    'SVC_rbf': SVC(
        random_state=42,
        class_weight='balanced',
        kernel='rbf',
        probability=True  # F1 계산용
    ),
    
    'SVC_linear': SVC(
        random_state=42,
        class_weight='balanced',
        kernel='linear',
        probability=True
    ),
    
    'LinearSVC': LinearSVC(
        random_state=42,
        class_weight='balanced',
        max_iter=2000
    ),
    
    # Neighbors
    'KNeighbors': KNeighborsClassifier(
        n_neighbors=5,
        weights='distance'  # 불균형 데이터에 도움
    ),
    
    # Discriminant Analysis
    'LinearDA': LinearDiscriminantAnalysis(),
    'QuadraticDA': QuadraticDiscriminantAnalysis(),
    
    # Neural Network
    'MLPClassifier': MLPClassifier(
        hidden_layer_sizes=(100,),
        random_state=42,
        max_iter=300
    ),
    
    # Additional Ensemble
    'Bagging': BaggingClassifier(
        n_estimators=10,
        random_state=42
    )
}

print(f"Total models to evaluate: {len(models)}")

# 결과 저장용
results = []

print(f"\n--- Model Evaluation (5-Fold Cross-Validation) ---")
print("Model                | CV F1 Score    | CV F1 Std | Training Time | Notes")
print("-" * 80)

# 각 모델 평가
for model_name, model in models.items():
    try:
        start_time = time.time()
        
        # Cross-validation으로 F1 점수 계산
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        
        training_time = time.time() - start_time
        
        mean_f1 = cv_scores.mean()
        std_f1 = cv_scores.std()
        
        result = {
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
            'model': model_name,
            'cv_f1_mean': 0.0,
            'cv_f1_std': 0.0,
            'training_time': 0.0,
            'cv_scores': [],
            'status': f'failed: {str(e)[:50]}'
        }
        
        results.append(result)
        print(f"{model_name:20} | Failed        | Failed    | {str(e)[:30]}")

# 결과 데이터프레임으로 정리
results_df = pd.DataFrame(results)
successful_results = results_df[results_df['status'] == 'success'].copy()
successful_results = successful_results.sort_values('cv_f1_mean', ascending=False)

print(f"\n=== Model Ranking (Successful Models Only) ===")
print("Rank | Model                | CV F1 Score    | Training Time | Efficiency Score")
print("-" * 80)

# 효율성 점수 계산 (F1 / log(time + 1))
for i, (_, row) in enumerate(successful_results.iterrows(), 1):
    efficiency = row['cv_f1_mean'] / np.log(row['training_time'] + 2)  # +2 to avoid log(1)=0
    print(f"{i:4d} | {row['model']:20} | {row['cv_f1_mean']:.4f}±{row['cv_f1_std']:.4f} | {row['training_time']:8.2f}s | {efficiency:.4f}")

print(f"\n=== Top 5 Models ===")
top5_models = successful_results.head(5)

for i, (_, row) in enumerate(top5_models.iterrows(), 1):
    print(f"{i}. {row['model']}: F1 = {row['cv_f1_mean']:.4f}±{row['cv_f1_std']:.4f}")

print(f"\n=== Failed Models ===")
failed_results = results_df[results_df['status'] != 'success']
if len(failed_results) > 0:
    for _, row in failed_results.iterrows():
        print(f"- {row['model']}: {row['status']}")
else:
    print("All models completed successfully!")

print(f"\n--- Detailed Analysis ---")

# 성능별 그룹화
high_performance = successful_results[successful_results['cv_f1_mean'] >= 0.45]
medium_performance = successful_results[
    (successful_results['cv_f1_mean'] >= 0.35) & 
    (successful_results['cv_f1_mean'] < 0.45)
]
low_performance = successful_results[successful_results['cv_f1_mean'] < 0.35]

print(f"High Performance (F1 >= 0.45): {len(high_performance)} models")
for _, row in high_performance.iterrows():
    print(f"  - {row['model']}: {row['cv_f1_mean']:.4f}")

print(f"\nMedium Performance (0.35 <= F1 < 0.45): {len(medium_performance)} models")
for _, row in medium_performance.iterrows():
    print(f"  - {row['model']}: {row['cv_f1_mean']:.4f}")

print(f"\nLow Performance (F1 < 0.35): {len(low_performance)} models")
for _, row in low_performance.iterrows():
    print(f"  - {row['model']}: {row['cv_f1_mean']:.4f}")

# 결과 저장
results_df.to_csv(RESULT_PATH + 'model_comparison_results.csv', index=False)
top5_models.to_csv(RESULT_PATH + 'top5_models.csv', index=False)

# 요약 통계 저장
summary_stats = {
    'total_models': len(models),
    'successful_models': len(successful_results),
    'failed_models': len(failed_results),
    'best_model': successful_results.iloc[0]['model'],
    'best_f1_score': float(successful_results.iloc[0]['cv_f1_mean']),
    'best_f1_std': float(successful_results.iloc[0]['cv_f1_std']),
    'high_performance_count': len(high_performance),
    'medium_performance_count': len(medium_performance),
    'low_performance_count': len(low_performance),
    'top5_models': top5_models['model'].tolist(),
    'top5_f1_scores': top5_models['cv_f1_mean'].tolist()
}

import json
with open(RESULT_PATH + 'comparison_summary.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)

print(f"\n=== Summary ===")
print(f"Total models evaluated: {len(models)}")
print(f"Successful models: {len(successful_results)}")
print(f"Best model: {summary_stats['best_model']} (F1: {summary_stats['best_f1_score']:.4f})")
print(f"High performance models (F1 >= 0.45): {len(high_performance)}")

print(f"\nFiles saved:")
print(f"- model_comparison_results.csv: 전체 결과")
print(f"- top5_models.csv: 상위 5개 모델")
print(f"- comparison_summary.json: 요약 통계")

print(f"\n모델 비교 완료!") 