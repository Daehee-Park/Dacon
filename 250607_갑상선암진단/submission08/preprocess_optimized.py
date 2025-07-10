import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

# 데이터 경로 설정
DATA_PATH = '../data/'
RESULT_PATH = './result/'

# 결과 폴더 생성
import os
os.makedirs(RESULT_PATH, exist_ok=True)

print("=== Optimized Feature Engineering Based on Core 27 Features ===")

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

print("\n--- 1. Core Feature Engineering (Based on Submission07 Insights) ---")

# 모든 데이터 합치기
all_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)

# 7차에서 발견한 핵심 특성들만 선별적으로 생성
print("Creating optimized core features...")

# 1. 핵심 호르몬 비율 (교호작용 최소화)
all_data['TSH_T4_ratio'] = all_data['TSH_Result'] / (all_data['T4_Result'] + 1e-5)
all_data['TSH_T3_ratio'] = all_data['TSH_Result'] / (all_data['T3_Result'] + 1e-5)
all_data['T4_T3_ratio'] = all_data['T4_Result'] / (all_data['T3_Result'] + 1e-5)

# 2. 호르몬 불균형 지표 (단순화)
all_data['hormone_imbalance'] = (
    np.abs(all_data['TSH_Result'] - all_data['TSH_Result'].median()) +
    np.abs(all_data['T4_Result'] - all_data['T4_Result'].median()) +
    np.abs(all_data['T3_Result'] - all_data['T3_Result'].median())
)

# 3. 위험 요인들을 binary로 변환 (교호작용 제거)
risk_factors = ['Family_Background', 'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Diabetes']
for feature in risk_factors:
    all_data[f'{feature}_binary'] = (
        (all_data[feature] == 'Positive') | 
        (all_data[feature] == 'Yes') |
        (all_data[feature] == 1)
    ).astype(int)

# 4. 핵심 위험 점수 (가중치 단순화)
all_data['risk_score'] = (
    all_data['Family_Background_binary'] * 2 +  # 가족력 중요도 조정
    all_data['Radiation_History_binary'] * 1.5 +
    all_data['Iodine_Deficiency_binary'] * 1 +
    all_data['Smoke_binary'] * 0.5
    # Diabetes는 중요도가 낮아 제거
)

# 5. 나이-결절 상호작용 (단순화)
all_data['age_nodule_interaction'] = all_data['Age'] * all_data['Nodule_Size']

print("\n--- 2. Feature Pooling & Simplification ---")

# 국가별 풀링 (유사한 패턴의 국가들 그룹화)
# 7차에서 중요했던 국가들: Country_BRA, Country_IND, Country_CHN, Country_NGA
country_mapping = {
    'BRA': 'Latin',
    'ARG': 'Latin', 
    'COL': 'Latin',
    'IND': 'Asian',
    'CHN': 'Asian',
    'THA': 'Asian',
    'NGA': 'African',
    'ETH': 'African',
    'ZAF': 'African'
}

# 기타 국가들은 'Other'로 풀링
all_data['country_group'] = all_data['Country'].map(country_mapping).fillna('Other')

# 인종별 풀링 (7차에서 중요했던 Race_ASN, Race_AFR, Race_CAU, Race_HSP, Race_MDE)
# 이미 의미있는 그룹이므로 유지하되, 소수 인종은 풀링
race_counts = all_data['Race'].value_counts()
major_races = race_counts[race_counts > 1000].index.tolist()
all_data['race_group'] = all_data['Race'].apply(lambda x: x if x in major_races else 'Other')

print("\n--- 3. Categorical Encoding ---")

# Label Encoding for binary features
binary_features = ['Gender', 'Family_Background', 'Radiation_History', 
                   'Iodine_Deficiency', 'Smoke', 'Diabetes']

label_encoders = {}
for feature in binary_features:
    le = LabelEncoder()
    all_data[feature] = le.fit_transform(all_data[feature])
    label_encoders[feature] = le

# One-hot encoding for pooled categorical features
categorical_features = ['country_group', 'race_group', 'Weight_Risk']
all_data_encoded = pd.get_dummies(all_data, columns=categorical_features, prefix=categorical_features)

print(f"Features after pooling and encoding: {all_data_encoded.shape[1]}")

print("\n--- 4. Feature Selection (Focus on Core Features) ---")

# 7차에서 발견한 핵심 특성들 우선 선택
core_features_base = [
    'Age', 'Gender', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
    'Family_Background', 'Radiation_History', 'Iodine_Deficiency', 'Smoke',
    'TSH_T4_ratio', 'TSH_T3_ratio', 'T4_T3_ratio', 'hormone_imbalance',
    'Family_Background_binary', 'risk_score', 'age_nodule_interaction'
]

# One-hot encoded 특성들 중 7차에서 중요했던 것들
country_features = [col for col in all_data_encoded.columns if 'country_group_' in col]
race_features = [col for col in all_data_encoded.columns if 'race_group_' in col]
weight_features = [col for col in all_data_encoded.columns if 'Weight_Risk_' in col]

# 핵심 특성 조합
core_features = core_features_base + country_features + race_features + weight_features

# 실제 존재하는 특성만 선택
available_features = [f for f in core_features if f in all_data_encoded.columns]
print(f"Available core features: {len(available_features)}")

# 추가 중요 특성 (7차 결과 기반)
additional_important = []
for col in all_data_encoded.columns:
    if any(keyword in col.lower() for keyword in ['asian', 'african', 'caucasian', 'brazilian', 'indian', 'chinese', 'nigerian']):
        if col not in available_features:
            additional_important.append(col)

final_features = available_features + additional_important
print(f"Final selected features: {len(final_features)}")

# 선택된 특성만 추출
all_data_selected = all_data_encoded[final_features]

print("\n--- 5. Advanced Feature Optimization ---")

# 5.1. 고도로 상관된 특성 제거 (교호작용 최소화)
correlation_matrix = all_data_selected.corr().abs()
upper_triangle = correlation_matrix.where(
    np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
)

# 0.95 이상 상관된 특성 쌍 찾기
high_corr_pairs = []
for column in upper_triangle.columns:
    high_corr_features = list(upper_triangle.index[upper_triangle[column] > 0.95])
    if high_corr_features:
        high_corr_pairs.extend([(column, feature) for feature in high_corr_features])

print(f"High correlation pairs (>0.95): {len(high_corr_pairs)}")

# 상관성이 높은 특성 중 하나 제거
features_to_remove = set()
for feat1, feat2 in high_corr_pairs:
    # 더 단순한 이름의 특성을 유지 (일반적으로 더 해석가능)
    if len(feat1) > len(feat2):
        features_to_remove.add(feat1)
    else:
        features_to_remove.add(feat2)

final_features_optimized = [f for f in final_features if f not in features_to_remove]
all_data_optimized = all_data_selected[final_features_optimized]

print(f"Features after correlation removal: {len(final_features_optimized)}")

print("\n--- 6. Feature Scaling ---")

# 수치형 특성만 스케일링
numerical_features = []
for col in all_data_optimized.columns:
    if all_data_optimized[col].dtype in ['int64', 'float64'] and all_data_optimized[col].nunique() > 10:
        numerical_features.append(col)

print(f"Numerical features to scale: {len(numerical_features)}")

scaler = StandardScaler()
all_data_final = all_data_optimized.copy()
if numerical_features:
    all_data_final[numerical_features] = scaler.fit_transform(all_data_optimized[numerical_features])

# Train/Test 분리
X_train_processed = all_data_final[:len(X_train)]
X_test_processed = all_data_final[len(X_train):]

print(f"Final train shape: {X_train_processed.shape}")
print(f"Final test shape: {X_test_processed.shape}")

print("\n--- 7. Saving Optimized Data ---")

# 최적화된 데이터 저장
X_train_processed.to_csv(RESULT_PATH + 'X_train_optimized.csv', index=False)
X_test_processed.to_csv(RESULT_PATH + 'X_test_optimized.csv', index=False)
y_train.to_csv(RESULT_PATH + 'y_train.csv', index=False)
test_ids.to_csv(RESULT_PATH + 'test_ids.csv', index=False)

# 특성 정보 저장
feature_info = pd.DataFrame({
    'feature': final_features_optimized,
    'type': ['numerical' if f in numerical_features else 'categorical' 
             for f in final_features_optimized]
})
feature_info.to_csv(RESULT_PATH + 'feature_info.csv', index=False)

# 제거된 특성 정보 저장
removed_features_info = {
    'high_correlation_removed': list(features_to_remove),
    'correlation_threshold': 0.95,
    'original_features': len(available_features),
    'final_features': len(final_features_optimized),
    'reduction_ratio': (len(available_features) - len(final_features_optimized)) / len(available_features)
}

import json
with open(RESULT_PATH + 'optimization_summary.json', 'w') as f:
    json.dump(removed_features_info, f, indent=2)

print(f"\n=== Feature Optimization Summary ===")
print(f"Original core features: {len(available_features)}")
print(f"High correlation removed: {len(features_to_remove)}")
print(f"Final optimized features: {len(final_features_optimized)}")
print(f"Reduction ratio: {removed_features_info['reduction_ratio']:.2%}")
print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(final_features_optimized) - len(numerical_features)}")

print(f"\nFinal feature list:")
for i, feat in enumerate(final_features_optimized, 1):
    print(f"{i:2d}. {feat}")

print("\n최적화된 전처리 완료!") 