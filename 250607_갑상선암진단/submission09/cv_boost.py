import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings

warnings.filterwarnings('ignore')

# 데이터 경로 설정
RESULT_PATH = './result/'

print("=== CV Score Boosting - 9th Attempt ===")
print("Goal: CV F1 Score > 0.487")

# 데이터 로드
X_train = pd.read_csv(RESULT_PATH + 'X_train_original.csv')
X_test = pd.read_csv(RESULT_PATH + 'X_test_original.csv')
y_train = pd.read_csv(RESULT_PATH + 'y_train.csv')['Cancer']
test_ids = pd.read_csv(RESULT_PATH + 'test_ids.csv')['ID']

print(f"Data shape: {X_train.shape}")
print(f"Target distribution: {y_train.value_counts().to_dict()}")
print(f"Positive rate: {y_train.mean():.4f}")

# Cross-validation 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 8차 시도 최적 하이퍼파라미터
best_params = {
    'LightGBM': {
        'n_estimators': 930,
        'learning_rate': 0.01293383399765917,
        'num_leaves': 89,
        'max_depth': 5,
        'feature_fraction': 0.7016713762093676,
        'bagging_fraction': 0.8292577848605198,
        'min_child_samples': 49,
        'reg_alpha': 2.6747437832317966,
        'reg_lambda': 5.402972689468923,
        'random_state': 42,
        'class_weight': 'balanced',
        'verbose': -1
    },
    'XGBoost': {
        'n_estimators': 950,
        'learning_rate': 0.011184420924446357,
        'max_depth': 5,
        'subsample': 0.7467143880908638,
        'colsample_bytree': 0.7737425087025644,
        'reg_alpha': 7.30084853191198,
        'reg_lambda': 6.355083933063061,
        'min_child_weight': 7,
        'gamma': 0.15590624136109432,
        'random_state': 42,
        'eval_metric': 'logloss',
        'scale_pos_weight': y_train.value_counts()[0]/y_train.value_counts()[1]
    },
    'CatBoost': {
        'n_estimators': 945,
        'learning_rate': 0.01607123851203988,
        'depth': 5,
        'l2_leaf_reg': 4.297256589643226,
        'random_strength': 4.56069984217036,
        'random_state': 42,
        'auto_class_weights': 'Balanced',
        'verbose': False
    }
}

print("\n--- Strategy 1: Fine-tuned Individual Models ---")

# 개별 모델들을 미세 조정하여 성능 향상 시도
models = {
    'LightGBM': lgb.LGBMClassifier(**best_params['LightGBM']),
    'XGBoost': xgb.XGBClassifier(**best_params['XGBoost']),
    'CatBoost': cb.CatBoostClassifier(**best_params['CatBoost'])
}

individual_results = {}

for name, model in models.items():
    print(f"\n{name} CV Evaluation:")
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    mean_score = cv_scores.mean()
    std_score = cv_scores.std()
    
    print(f"CV F1: {mean_score:.4f} ± {std_score:.4f}")
    print(f"CV Scores: {[f'{score:.4f}' for score in cv_scores]}")
    
    individual_results[name] = {
        'cv_scores': cv_scores,
        'mean': mean_score,
        'std': std_score
    }

print(f"\n--- Strategy 2: Ensemble Methods ---")

# 1. Voting Classifier (Soft Voting)
print(f"\n1. Soft Voting Ensemble:")
voting_clf = VotingClassifier(
    estimators=[
        ('lgb', lgb.LGBMClassifier(**best_params['LightGBM'])),
        ('xgb', xgb.XGBClassifier(**best_params['XGBoost'])),
        ('cat', cb.CatBoostClassifier(**best_params['CatBoost']))
    ],
    voting='soft'
)

voting_scores = cross_val_score(voting_clf, X_train, y_train, cv=cv, scoring='f1', n_jobs=1)
voting_mean = voting_scores.mean()
voting_std = voting_scores.std()

print(f"Voting CV F1: {voting_mean:.4f} ± {voting_std:.4f}")
print(f"Voting CV Scores: {[f'{score:.4f}' for score in voting_scores]}")

# 2. Stacking Classifier
print(f"\n2. Stacking Ensemble:")
stacking_clf = StackingClassifier(
    estimators=[
        ('lgb', lgb.LGBMClassifier(**best_params['LightGBM'])),
        ('xgb', xgb.XGBClassifier(**best_params['XGBoost'])),
        ('cat', cb.CatBoostClassifier(**best_params['CatBoost']))
    ],
    final_estimator=LogisticRegression(class_weight='balanced', random_state=42),
    cv=3
)

stacking_scores = cross_val_score(stacking_clf, X_train, y_train, cv=cv, scoring='f1', n_jobs=1)
stacking_mean = stacking_scores.mean()
stacking_std = stacking_scores.std()

print(f"Stacking CV F1: {stacking_mean:.4f} ± {stacking_std:.4f}")
print(f"Stacking CV Scores: {[f'{score:.4f}' for score in stacking_scores]}")

print(f"\n--- Strategy 3: Feature Engineering Enhancement ---")

# 특성 상호작용 생성
def create_interaction_features(X):
    X_new = X.copy()
    
    # Age 관련 상호작용
    if 'Age' in X_new.columns:
        # Age와 다른 특성들의 상호작용
        for col in ['T4_Result', 'T3_Result']:
            if col in X_new.columns:
                X_new[f'Age_{col}_interaction'] = X_new['Age'] * X_new[col]
    
    # 호르몬 수치 관련 상호작용
    if 'T4_Result' in X_new.columns and 'T3_Result' in X_new.columns:
        X_new['T4_T3_ratio'] = X_new['T4_Result'] / (X_new['T3_Result'] + 1e-8)
        X_new['T4_T3_sum'] = X_new['T4_Result'] + X_new['T3_Result']
        X_new['T4_T3_diff'] = X_new['T4_Result'] - X_new['T3_Result']
    
    # Age 구간별 특성
    if 'Age' in X_new.columns:
        X_new['Age_binned'] = pd.cut(X_new['Age'], bins=5, labels=[0,1,2,3,4]).astype(int)
    
    return X_new

print(f"\n3. Feature Engineering with Interactions:")
X_train_enhanced = create_interaction_features(X_train)
X_test_enhanced = create_interaction_features(X_test)

print(f"Original features: {X_train.shape[1]}")
print(f"Enhanced features: {X_train_enhanced.shape[1]}")

# Enhanced 데이터로 최고 개별 모델 테스트
best_individual = max(individual_results.items(), key=lambda x: x[1]['mean'])
best_model_name = best_individual[0]
print(f"Best individual model: {best_model_name} (CV: {best_individual[1]['mean']:.4f})")

if best_model_name == 'LightGBM':
    enhanced_model = lgb.LGBMClassifier(**best_params['LightGBM'])
elif best_model_name == 'XGBoost':
    enhanced_model = xgb.XGBClassifier(**best_params['XGBoost'])
else:
    enhanced_model = cb.CatBoostClassifier(**best_params['CatBoost'])

enhanced_scores = cross_val_score(enhanced_model, X_train_enhanced, y_train, cv=cv, scoring='f1', n_jobs=-1)
enhanced_mean = enhanced_scores.mean()
enhanced_std = enhanced_scores.std()

print(f"Enhanced {best_model_name} CV F1: {enhanced_mean:.4f} ± {enhanced_std:.4f}")
print(f"Enhanced CV Scores: {[f'{score:.4f}' for score in enhanced_scores]}")

print(f"\n--- Strategy 4: Weighted Ensemble ---")

# 성능 기반 가중 앙상블
weights = np.array([individual_results[name]['mean'] for name in ['LightGBM', 'XGBoost', 'CatBoost']])
weights = weights / weights.sum()

print(f"Model weights based on CV performance:")
for i, name in enumerate(['LightGBM', 'XGBoost', 'CatBoost']):
    print(f"- {name}: {weights[i]:.3f}")

def weighted_ensemble_predict_proba(models, X, weights):
    predictions = []
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        pred_proba = model.predict_proba(X)[:, 1]
        predictions.append(pred_proba * weights[i])
    return np.sum(predictions, axis=0)

# Weighted ensemble CV 평가
weighted_cv_scores = []

for train_idx, val_idx in cv.split(X_train, y_train):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # 각 모델 학습
    trained_models = {}
    for name, model in models.items():
        trained_models[name] = model.fit(X_tr, y_tr)
    
    # Weighted ensemble 예측
    val_proba = weighted_ensemble_predict_proba(trained_models, X_val, weights)
    val_pred = (val_proba >= 0.5).astype(int)
    
    # F1 score 계산
    f1 = f1_score(y_val, val_pred)
    weighted_cv_scores.append(f1)

weighted_mean = np.mean(weighted_cv_scores)
weighted_std = np.std(weighted_cv_scores)

print(f"Weighted Ensemble CV F1: {weighted_mean:.4f} ± {weighted_std:.4f}")
print(f"Weighted CV Scores: {[f'{score:.4f}' for score in weighted_cv_scores]}")

print(f"\n=== Results Summary ===")

results_summary = {
    'Individual Models': {
        'LightGBM': individual_results['LightGBM']['mean'],
        'XGBoost': individual_results['XGBoost']['mean'],
        'CatBoost': individual_results['CatBoost']['mean']
    },
    'Ensemble Methods': {
        'Soft Voting': voting_mean,
        'Stacking': stacking_mean,
        'Weighted Ensemble': weighted_mean
    },
    'Feature Engineering': {
        f'Enhanced {best_model_name}': enhanced_mean
    }
}

# 모든 결과를 성능순으로 정렬
all_results = {}
for category, methods in results_summary.items():
    for method, score in methods.items():
        all_results[f"{category} - {method}"] = score

sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)

print(f"\nRanking (CV F1 Score):")
for i, (method, score) in enumerate(sorted_results, 1):
    goal_check = "✅" if score > 0.487 else "❌"
    print(f"{i}. {method}: {score:.4f} {goal_check}")

# 최고 성능 모델 선택 및 최종 예측
best_method, best_score = sorted_results[0]
print(f"\n--- Final Model Selection ---")
print(f"Best method: {best_method}")
print(f"Best CV F1: {best_score:.4f}")

# 최고 성능 방법에 따라 최종 모델 학습 및 예측
if "Enhanced" in best_method:
    print("Using enhanced features for final prediction")
    final_X_train = X_train_enhanced
    final_X_test = X_test_enhanced
    final_model = enhanced_model
else:
    final_X_train = X_train
    final_X_test = X_test

if "Voting" in best_method:
    final_model = voting_clf
elif "Stacking" in best_method:
    final_model = stacking_clf
elif "Weighted" in best_method:
    # Weighted ensemble의 경우 별도 처리
    final_pred_proba = weighted_ensemble_predict_proba(models, final_X_test, weights)
    final_pred = (final_pred_proba >= 0.5).astype(int)
else:
    # Individual model
    model_name = best_method.split(' - ')[1]
    if 'LightGBM' in model_name:
        final_model = lgb.LGBMClassifier(**best_params['LightGBM'])
    elif 'XGBoost' in model_name:
        final_model = xgb.XGBClassifier(**best_params['XGBoost'])
    else:
        final_model = cb.CatBoostClassifier(**best_params['CatBoost'])

# Weighted ensemble가 아닌 경우에만 모델 학습 및 예측
if "Weighted" not in best_method:
    final_model.fit(final_X_train, y_train)
    final_pred_proba = final_model.predict_proba(final_X_test)[:, 1]
    final_pred = final_model.predict(final_X_test)

# 임계값 최적화 (Train set에서)
final_model_for_threshold = final_model if "Weighted" not in best_method else models['XGBoost']
final_model_for_threshold.fit(final_X_train, y_train)
train_proba = final_model_for_threshold.predict_proba(final_X_train)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.01)
best_threshold = 0.5
best_f1_thresh = 0

for threshold in thresholds:
    pred_thresh = (train_proba >= threshold).astype(int)
    f1_thresh = f1_score(y_train, pred_thresh)
    if f1_thresh > best_f1_thresh:
        best_f1_thresh = f1_thresh
        best_threshold = threshold

print(f"\nThreshold optimization:")
print(f"Best threshold: {best_threshold:.3f}")
print(f"F1 with best threshold: {best_f1_thresh:.4f}")

# 최적 임계값으로 최종 예측 재조정
final_pred_optimized = (final_pred_proba >= best_threshold).astype(int)

print(f"\nFinal predictions:")
print(f"Predicted positive rate: {final_pred_optimized.mean():.4f}")
print(f"Prediction distribution: {pd.Series(final_pred_optimized).value_counts().to_dict()}")

# 제출 파일 생성
submission_df = pd.DataFrame({
    'ID': test_ids,
    'Cancer': final_pred_optimized
})

submission_df.to_csv(RESULT_PATH + 'submission09.csv', index=False)

# 결과 저장
import json

final_summary = {
    'best_method': best_method,
    'best_cv_f1': float(best_score),
    'goal_achieved': bool(best_score > 0.487),
    'best_threshold': float(best_threshold),
    'threshold_f1': float(best_f1_thresh),
    'predicted_positive_rate': float(final_pred_optimized.mean()),
    'all_results': {k: float(v) for k, v in all_results.items()},
    'feature_count': int(final_X_train.shape[1])
}

with open(RESULT_PATH + 'cv_boost_summary.json', 'w') as f:
    json.dump(final_summary, f, indent=2)

print(f"\n=== Final Summary ===")
print(f"Best method: {best_method}")
print(f"Best CV F1: {best_score:.4f}")
goal_status = "✅ ACHIEVED" if best_score > 0.487 else "❌ NOT YET"
print(f"Goal (CV F1 > 0.487): {goal_status}")

print(f"\nFiles saved:")
print(f"- submission09.csv: 제출 파일")
print(f"- cv_boost_summary.json: 9차 시도 요약")

print(f"\n9차 CV Boosting 완료!") 