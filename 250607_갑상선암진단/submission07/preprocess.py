import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# 데이터 경로 설정
DATA_PATH = '../data/'
RESULT_PATH = './result/'

# 결과 폴더 생성
import os
os.makedirs(RESULT_PATH, exist_ok=True)

print("=== Feature Selection & Dimensionality Reduction Approach ===")

# 데이터 로드
train_df = pd.read_csv(DATA_PATH + 'train.csv')
test_df = pd.read_csv(DATA_PATH + 'test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Target 분리
y_train = train_df['Cancer']
X_train = train_df.drop(['ID', 'Cancer'], axis=1)
X_test = test_df.drop(['ID'], axis=1)

# Test ID 저장
test_ids = test_df['ID']

print(f"Target distribution:")
print(y_train.value_counts())
print(f"Positive rate: {y_train.mean():.4f}")

print("\n--- 1. Basic Feature Engineering ---")

# 모든 데이터 합치기 (전처리용)
all_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)

# 핵심 의료 특성만 생성 (너무 많이 만들지 않고 선별적으로)
print("Creating key medical features...")

# 갑상선 호르몬 비율 (가장 중요한 특성들)
all_data['TSH_T4_ratio'] = all_data['TSH_Result'] / (all_data['T4_Result'] + 1e-5)
all_data['TSH_T3_ratio'] = all_data['TSH_Result'] / (all_data['T3_Result'] + 1e-5)
all_data['T4_T3_ratio'] = all_data['T4_Result'] / (all_data['T3_Result'] + 1e-5)

# 호르몬 불균형 지표
all_data['hormone_imbalance'] = (
    np.abs(all_data['TSH_Result'] - all_data['TSH_Result'].median()) +
    np.abs(all_data['T4_Result'] - all_data['T4_Result'].median()) +
    np.abs(all_data['T3_Result'] - all_data['T3_Result'].median())
)

# 위험 요인들을 binary로 변환
risk_factors = ['Family_Background', 'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Diabetes']
for feature in risk_factors:
    all_data[f'{feature}_binary'] = (
        (all_data[feature] == 'Positive') | 
        (all_data[feature] == 'Yes') |
        (all_data[feature] == 1)
    ).astype(int)

# 복합 위험 점수
all_data['risk_score'] = (
    all_data['Family_Background_binary'] * 3 +
    all_data['Radiation_History_binary'] * 2 +
    all_data['Iodine_Deficiency_binary'] * 1.5 +
    all_data['Smoke_binary'] * 1 +
    all_data['Diabetes_binary'] * 0.5
)

# 나이-결절 크기 상호작용
all_data['age_nodule_interaction'] = all_data['Age'] * all_data['Nodule_Size']

print("\n--- 2. Categorical Encoding ---")

# Label Encoding for binary features
binary_features = ['Gender', 'Family_Background', 'Radiation_History', 
                   'Iodine_Deficiency', 'Smoke', 'Diabetes']

label_encoders = {}
for feature in binary_features:
    le = LabelEncoder()
    all_data[feature] = le.fit_transform(all_data[feature])
    label_encoders[feature] = le

# One-hot encoding for multi-class features (간단하게)
categorical_features = ['Country', 'Race', 'Weight_Risk']
all_data_encoded = pd.get_dummies(all_data, columns=categorical_features, prefix=categorical_features)

print(f"Features after basic encoding: {all_data_encoded.shape[1]}")

print("\n--- 3. Feature Scaling ---")

# 수치형 특성 스케일링
numerical_features = [col for col in all_data_encoded.columns 
                     if all_data_encoded[col].dtype in ['int64', 'float64'] 
                     and not col.endswith('_binary')]

scaler = StandardScaler()
all_data_encoded[numerical_features] = scaler.fit_transform(all_data_encoded[numerical_features])

# Train/Test 분리
X_train_processed = all_data_encoded[:len(X_train)]
X_test_processed = all_data_encoded[len(X_train):]

print(f"Before feature selection - Train shape: {X_train_processed.shape}")

print("\n--- 4. Feature Selection & Importance Analysis ---")

# 4.1. Random Forest Feature Importance
print("4.1. Random Forest Feature Importance Analysis...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_processed, y_train)

# 특성 중요도 저장
feature_importance_rf = pd.DataFrame({
    'feature': X_train_processed.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"Top 20 important features (Random Forest):")
print(feature_importance_rf.head(20))

# 4.2. LightGBM Feature Importance
print("\n4.2. LightGBM Feature Importance Analysis...")
lgb_train = lgb.Dataset(X_train_processed, y_train)
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'verbose': -1,
    'random_state': 42
}

lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100)
feature_importance_lgb = pd.DataFrame({
    'feature': X_train_processed.columns,
    'importance': lgb_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print(f"Top 20 important features (LightGBM):")
print(feature_importance_lgb.head(20))

# 4.3. Statistical Feature Selection
print("\n4.3. Statistical Feature Selection...")

# F-test based selection
selector_f = SelectKBest(score_func=f_classif, k=30)
X_train_f = selector_f.fit_transform(X_train_processed, y_train)
selected_features_f = X_train_processed.columns[selector_f.get_support()].tolist()

print(f"F-test selected features (top 30): {len(selected_features_f)}")
print(selected_features_f[:15])

# Mutual Information based selection
selector_mi = SelectKBest(score_func=mutual_info_classif, k=30)
X_train_mi = selector_mi.fit_transform(X_train_processed, y_train)
selected_features_mi = X_train_processed.columns[selector_mi.get_support()].tolist()

print(f"\nMutual Info selected features (top 30): {len(selected_features_mi)}")
print(selected_features_mi[:15])

print("\n--- 5. Feature Selection Strategies ---")

# 5.1. Top features by Random Forest importance (상위 30개)
top_rf_features = feature_importance_rf.head(30)['feature'].tolist()
X_train_rf_top = X_train_processed[top_rf_features]
X_test_rf_top = X_test_processed[top_rf_features]

# 5.2. Top features by LightGBM importance (상위 30개)
top_lgb_features = feature_importance_lgb.head(30)['feature'].tolist()
X_train_lgb_top = X_train_processed[top_lgb_features]
X_test_lgb_top = X_test_processed[top_lgb_features]

# 5.3. 교집합 특성 (RF와 LGB 공통 중요 특성)
common_features = list(set(top_rf_features) & set(top_lgb_features))
print(f"Common important features: {len(common_features)}")
print(common_features)

if len(common_features) >= 20:
    X_train_common = X_train_processed[common_features]
    X_test_common = X_test_processed[common_features]
else:
    # 교집합이 적으면 합집합에서 상위 25개 선택
    union_features = list(set(top_rf_features[:15] + top_lgb_features[:15]))
    X_train_common = X_train_processed[union_features]
    X_test_common = X_test_processed[union_features]
    common_features = union_features

# 5.4. F-test 기반 선택
X_test_f = selector_f.transform(X_test_processed)

# 5.5. Mutual Information 기반 선택
X_test_mi = selector_mi.transform(X_test_processed)

print("\n--- 6. Dimensionality Reduction ---")

# 6.1. PCA (상위 20차원)
print("6.1. PCA Analysis...")
pca = PCA(n_components=20, random_state=42)
X_train_pca = pca.fit_transform(X_train_processed)
X_test_pca = pca.transform(X_test_processed)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"PCA explained variance ratio (first 10): {explained_variance_ratio[:10]}")
print(f"Cumulative variance (20 components): {cumulative_variance[-1]:.4f}")

# 6.2. PCA with different components
pca_15 = PCA(n_components=15, random_state=42)
X_train_pca_15 = pca_15.fit_transform(X_train_processed)
X_test_pca_15 = pca_15.transform(X_test_processed)

print("\n--- 7. Saving Processed Data ---")

# 다양한 feature selection 결과 저장
datasets = {
    'original': (X_train_processed, X_test_processed),
    'rf_top30': (X_train_rf_top, X_test_rf_top),
    'lgb_top30': (X_train_lgb_top, X_test_lgb_top),
    'common_features': (X_train_common, X_test_common),
    'f_test_30': (pd.DataFrame(X_train_f), pd.DataFrame(X_test_f)),
    'mutual_info_30': (pd.DataFrame(X_train_mi), pd.DataFrame(X_test_mi)),
    'pca_20': (pd.DataFrame(X_train_pca), pd.DataFrame(X_test_pca)),
    'pca_15': (pd.DataFrame(X_train_pca_15), pd.DataFrame(X_test_pca_15))
}

for name, (X_tr, X_te) in datasets.items():
    X_tr.to_csv(RESULT_PATH + f'X_train_{name}.csv', index=False)
    X_te.to_csv(RESULT_PATH + f'X_test_{name}.csv', index=False)
    print(f"{name}: {X_tr.shape}")

# Target과 test_ids는 공통
y_train.to_csv(RESULT_PATH + 'y_train.csv', index=False)
test_ids.to_csv(RESULT_PATH + 'test_ids.csv', index=False)

# Feature importance 저장
feature_importance_rf.to_csv(RESULT_PATH + 'feature_importance_rf.csv', index=False)
feature_importance_lgb.to_csv(RESULT_PATH + 'feature_importance_lgb.csv', index=False)

# Selected features 저장
pd.DataFrame({'selected_features_f': selected_features_f}).to_csv(RESULT_PATH + 'selected_features_f.csv', index=False)
pd.DataFrame({'selected_features_mi': selected_features_mi}).to_csv(RESULT_PATH + 'selected_features_mi.csv', index=False)
pd.DataFrame({'common_features': common_features}).to_csv(RESULT_PATH + 'common_features.csv', index=False)

print(f"\n=== Feature Selection Summary ===")
print(f"Original features: {X_train_processed.shape[1]}")
print(f"RF top 30: {len(top_rf_features)}")
print(f"LGB top 30: {len(top_lgb_features)}")  
print(f"Common features: {len(common_features)}")
print(f"F-test selected: {len(selected_features_f)}")
print(f"Mutual info selected: {len(selected_features_mi)}")
print(f"PCA 20 components: 20")
print(f"PCA 15 components: 15")

print("\n전처리 및 특성 선택 완료!") 