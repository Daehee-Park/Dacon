import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings

warnings.filterwarnings('ignore')

# 데이터 경로 설정
RESULT_PATH = './result/'

print("=== Advanced CV Boosting - 9th Attempt (Round 2) ===")
print("Target: CV F1 Score > 0.487 (MUST ACHIEVE!)")

# 데이터 로드
X_train = pd.read_csv(RESULT_PATH + 'X_train_original.csv')
X_test = pd.read_csv(RESULT_PATH + 'X_test_original.csv')
y_train = pd.read_csv(RESULT_PATH + 'y_train.csv')['Cancer']
test_ids = pd.read_csv(RESULT_PATH + 'test_ids.csv')['ID']

print(f"Data shape: {X_train.shape}")
print(f"Positive rate: {y_train.mean():.4f}")

# Cross-validation 설정
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 8차 최적 하이퍼파라미터
best_params = {
    'LightGBM': {
        'n_estimators': 330,
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
        'n_estimators': 150,
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
        'n_estimators': 345,
        'learning_rate': 0.01607123851203988,
        'depth': 5,
        'l2_leaf_reg': 4.297256589643226,
        'random_strength': 4.56069984217036,
        'random_state': 42,
        'auto_class_weights': 'Balanced',
        'verbose': False
    }
}

print("\n=== Strategy 1: Advanced Feature Engineering ===")

def create_advanced_features(X):
    X_new = X.copy()
    
    # 1. 더 복잡한 상호작용 특성
    numeric_cols = X_new.select_dtypes(include=[np.number]).columns
    
    if 'Age' in X_new.columns:
        # Age 관련 고급 특성
        X_new['Age_squared'] = X_new['Age'] ** 2
        X_new['Age_log'] = np.log1p(X_new['Age'])
        X_new['Age_sqrt'] = np.sqrt(X_new['Age'])
        
        # Age 구간별 더 세분화
        X_new['Age_group'] = pd.cut(X_new['Age'], bins=[0, 30, 50, 70, 100], labels=[0,1,2,3])
        X_new['Age_group'] = X_new['Age_group'].fillna(0).astype(int)
    
    # 2. 호르몬 수치 고급 특성
    if 'T4_Result' in X_new.columns and 'T3_Result' in X_new.columns:
        X_new['T4_T3_ratio'] = X_new['T4_Result'] / (X_new['T3_Result'] + 1e-8)
        X_new['T4_T3_product'] = X_new['T4_Result'] * X_new['T3_Result']
        X_new['T4_T3_harmonic_mean'] = 2 * X_new['T4_Result'] * X_new['T3_Result'] / (X_new['T4_Result'] + X_new['T3_Result'] + 1e-8)
        
        # 호르몬 수치 정상화 여부
        X_new['T4_normal'] = ((X_new['T4_Result'] >= 0.8) & (X_new['T4_Result'] <= 1.8)).astype(int)
        X_new['T3_normal'] = ((X_new['T3_Result'] >= 0.8) & (X_new['T3_Result'] <= 2.0)).astype(int)
        X_new['hormones_both_normal'] = X_new['T4_normal'] * X_new['T3_normal']
    
    # 3. 통계적 특성
    for i in range(len(numeric_cols)):
        for j in range(i+1, len(numeric_cols)):
            col1, col2 = numeric_cols[i], numeric_cols[j]
            if col1 != col2:
                X_new[f'{col1}_{col2}_sum'] = X_new[col1] + X_new[col2]
                X_new[f'{col1}_{col2}_diff'] = X_new[col1] - X_new[col2]
                X_new[f'{col1}_{col2}_ratio'] = X_new[col1] / (X_new[col2] + 1e-8)
    
    # 4. 폴리노미얼 특성 (일부)
    from sklearn.preprocessing import PolynomialFeatures
    
    # 중요한 특성들만 선택해서 폴리노미얼 적용
    important_cols = ['Age', 'T4_Result', 'T3_Result'] if all(c in X_new.columns for c in ['Age', 'T4_Result', 'T3_Result']) else numeric_cols[:3]
    if len(important_cols) > 0:
        poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
        poly_features = poly.fit_transform(X_new[important_cols])
        poly_feature_names = poly.get_feature_names_out(important_cols)
        
        for i, name in enumerate(poly_feature_names):
            if name not in X_new.columns:  # 이미 존재하지 않는 특성만 추가
                X_new[f'poly_{name}'] = poly_features[:, i]
    
    # NaN 값 처리
    X_new = X_new.fillna(0)
    
    # Infinite 값 처리
    X_new = X_new.replace([np.inf, -np.inf], 0)
    
    return X_new

print("Creating advanced engineered features...")
X_train_advanced = create_advanced_features(X_train)
X_test_advanced = create_advanced_features(X_test)

print(f"Original features: {X_train.shape[1]}")
print(f"Advanced features: {X_train_advanced.shape[1]}")

print("\n=== Strategy 2: Feature Selection ===")

# 다양한 특성 선택 방법 시도
def select_features_multiple_methods(X, y, n_features=50):
    results = {}
    
    # 1. Mutual Information
    mi_selector = SelectKBest(mutual_info_classif, k=n_features)
    mi_selector.fit(X, y)
    mi_features = X.columns[mi_selector.get_support()]
    results['mutual_info'] = mi_features
    
    # 2. F-test
    f_selector = SelectKBest(f_classif, k=n_features)
    f_selector.fit(X, y)
    f_features = X.columns[f_selector.get_support()]
    results['f_test'] = f_features
    
    # 3. Random Forest 중요도
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    rf_features = rf_importance.nlargest(n_features).index
    results['random_forest'] = rf_features
    
    # 4. LightGBM 중요도
    lgb_model = lgb.LGBMClassifier(**best_params['LightGBM'])
    lgb_model.fit(X, y)
    lgb_importance = pd.Series(lgb_model.feature_importances_, index=X.columns)
    lgb_features = lgb_importance.nlargest(n_features).index
    results['lightgbm'] = lgb_features
    
    return results

feature_selections = select_features_multiple_methods(X_train_advanced, y_train, n_features=40)

print("\n=== Strategy 3: Neural Network ===")

# 신경망 모델 추가
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_advanced)
X_test_scaled = scaler.transform(X_test_advanced)

nn_model = MLPClassifier(
    hidden_layer_sizes=(100, 50, 20),
    activation='relu',
    solver='adam',
    alpha=0.01,
    learning_rate='adaptive',
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

nn_scores = cross_val_score(nn_model, X_train_scaled, y_train, cv=cv, scoring='f1', n_jobs=-1)
nn_mean = nn_scores.mean()
nn_std = nn_scores.std()

print(f"Neural Network CV F1: {nn_mean:.4f} ± {nn_std:.4f}")
print(f"NN CV Scores: {[f'{score:.4f}' for score in nn_scores]}")

print("\n=== Strategy 4: PCA + Models ===")

# PCA를 적용한 차원 축소
pca_components = [20, 30, 40]
pca_results = {}

for n_comp in pca_components:
    pca = PCA(n_components=n_comp, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    # PCA 데이터로 XGBoost 테스트
    xgb_pca = xgb.XGBClassifier(**best_params['XGBoost'])
    pca_scores = cross_val_score(xgb_pca, X_train_pca, y_train, cv=cv, scoring='f1', n_jobs=-1)
    pca_mean = pca_scores.mean()
    
    pca_results[f'PCA_{n_comp}'] = pca_mean
    print(f"PCA {n_comp} components + XGBoost CV F1: {pca_mean:.4f}")

print("\n=== Strategy 5: Multi-level Stacking ===")

# 더 복잡한 스태킹 구조
level1_models = [
    ('lgb', lgb.LGBMClassifier(**best_params['LightGBM'])),
    ('xgb', xgb.XGBClassifier(**best_params['XGBoost'])),
    ('cat', cb.CatBoostClassifier(**best_params['CatBoost'])),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
    ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
]

multilevel_stacking = StackingClassifier(
    estimators=level1_models,
    final_estimator=LogisticRegression(class_weight='balanced', random_state=42, C=0.1),
    cv=3
)

stacking_scores = cross_val_score(multilevel_stacking, X_train_advanced, y_train, cv=cv, scoring='f1', n_jobs=1)
stacking_mean = stacking_scores.mean()
stacking_std = stacking_scores.std()

print(f"Multi-level Stacking CV F1: {stacking_mean:.4f} ± {stacking_std:.4f}")
print(f"Stacking CV Scores: {[f'{score:.4f}' for score in stacking_scores]}")

print("\n=== Strategy 6: Feature Selection + Best Model ===")

# 각 특성 선택 방법으로 최고 모델 테스트
selection_results = {}

for method, features in feature_selections.items():
    X_selected = X_train_advanced[features]
    
    # XGBoost로 테스트 (기존 최고 모델)
    xgb_selected = xgb.XGBClassifier(**best_params['XGBoost'])
    selected_scores = cross_val_score(xgb_selected, X_selected, y_train, cv=cv, scoring='f1', n_jobs=-1)
    selected_mean = selected_scores.mean()
    
    selection_results[f'Selected_{method}'] = selected_mean
    print(f"Feature selection ({method}) + XGBoost CV F1: {selected_mean:.4f}")

print("\n=== Advanced Results Summary ===")

# 모든 결과 수집
all_advanced_results = {
    'Neural Network': nn_mean,
    'Multi-level Stacking': stacking_mean,
    **pca_results,
    **selection_results
}

# 기존 모델들도 포함
all_advanced_results['XGBoost (baseline)'] = 0.4870

# 결과 정렬
sorted_advanced = sorted(all_advanced_results.items(), key=lambda x: x[1], reverse=True)

print(f"\nAdvanced Methods Ranking:")
for i, (method, score) in enumerate(sorted_advanced, 1):
    goal_check = "✅" if score > 0.487 else "❌"
    improvement = f"(+{((score - 0.4870) / 0.4870 * 100):+.2f}%)" if score != 0.4870 else "(baseline)"
    print(f"{i}. {method}: {score:.4f} {goal_check} {improvement}")

# 최고 성능 방법 선택
best_advanced_method, best_advanced_score = sorted_advanced[0]
print(f"\n--- Final Advanced Model Selection ---")
print(f"Best advanced method: {best_advanced_method}")
print(f"Best advanced CV F1: {best_advanced_score:.4f}")

# 목표 달성 여부 확인
goal_achieved = best_advanced_score > 0.487
print(f"Goal (CV F1 > 0.487): {'✅ ACHIEVED!' if goal_achieved else '❌ NOT YET'}")

# 최고 성능 모델로 최종 예측
if 'Neural Network' in best_advanced_method:
    print("Using Neural Network for final prediction")
    final_model = MLPClassifier(
        hidden_layer_sizes=(100, 50, 20),
        activation='relu',
        solver='adam',
        alpha=0.01,
        learning_rate='adaptive',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    final_X_train = X_train_scaled
    final_X_test = X_test_scaled
    
elif 'Stacking' in best_advanced_method:
    print("Using Multi-level Stacking for final prediction")
    final_model = multilevel_stacking
    final_X_train = X_train_advanced
    final_X_test = X_test_advanced
    
elif 'PCA' in best_advanced_method:
    n_comp = int(best_advanced_method.split('_')[1])
    print(f"Using PCA {n_comp} + XGBoost for final prediction")
    pca = PCA(n_components=n_comp, random_state=42)
    final_X_train = pca.fit_transform(X_train_scaled)
    final_X_test = pca.transform(X_test_scaled)
    final_model = xgb.XGBClassifier(**best_params['XGBoost'])
    
elif 'Selected' in best_advanced_method:
    method = best_advanced_method.split('_')[1]
    selected_features = feature_selections[method]
    print(f"Using feature selection ({method}) + XGBoost for final prediction")
    final_X_train = X_train_advanced[selected_features]
    final_X_test = X_test_advanced[selected_features]
    final_model = xgb.XGBClassifier(**best_params['XGBoost'])
    
else:
    # XGBoost baseline
    final_model = xgb.XGBClassifier(**best_params['XGBoost'])
    final_X_train = X_train_advanced
    final_X_test = X_test_advanced

# 최종 모델 학습 및 예측
final_model.fit(final_X_train, y_train)
final_pred_proba = final_model.predict_proba(final_X_test)[:, 1]

# 임계값 최적화
train_proba = final_model.predict_proba(final_X_train)[:, 1]

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

# 최적 임계값으로 최종 예측
final_pred = (final_pred_proba >= best_threshold).astype(int)

print(f"\nFinal predictions:")
print(f"Predicted positive rate: {final_pred.mean():.4f}")
print(f"Prediction distribution: {pd.Series(final_pred).value_counts().to_dict()}")

# 제출 파일 생성
submission_df = pd.DataFrame({
    'ID': test_ids,
    'Cancer': final_pred
})

submission_df.to_csv(RESULT_PATH + 'submission09_advanced.csv', index=False)

# 결과 저장
import json

advanced_summary = {
    'best_method': best_advanced_method,
    'best_cv_f1': float(best_advanced_score),
    'goal_achieved': goal_achieved,
    'improvement_over_baseline': float((best_advanced_score - 0.4870) / 0.4870 * 100),
    'best_threshold': float(best_threshold),
    'threshold_f1': float(best_f1_thresh),
    'predicted_positive_rate': float(final_pred.mean()),
    'all_advanced_results': {k: float(v) for k, v in all_advanced_results.items()},
    'feature_count': int(final_X_train.shape[1] if hasattr(final_X_train, 'shape') else len(final_X_train[0]))
}

with open(RESULT_PATH + 'advanced_cv_summary.json', 'w') as f:
    json.dump(advanced_summary, f, indent=2)

print(f"\n=== Advanced Final Summary ===")
print(f"Best method: {best_advanced_method}")
print(f"Best CV F1: {best_advanced_score:.4f}")
print(f"Improvement: {((best_advanced_score - 0.4870) / 0.4870 * 100):+.2f}%")
goal_status = "✅ MISSION ACCOMPLISHED!" if goal_achieved else "❌ CONTINUE FIGHTING!"
print(f"Goal Achievement: {goal_status}")

print(f"\nFiles saved:")
print(f"- submission09_advanced.csv: 제출 파일")
print(f"- advanced_cv_summary.json: 고급 시도 요약")

print(f"\n고급 CV Boosting 완료!") 