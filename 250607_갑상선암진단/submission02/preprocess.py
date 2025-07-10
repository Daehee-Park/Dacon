import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# 데이터 경로 설정
DATA_PATH = '../data/'
RESULT_PATH = './result/'

# 결과 폴더 생성
import os
os.makedirs(RESULT_PATH, exist_ok=True)

print("=== Enhanced Preprocessing with Medical Domain Knowledge ===")

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

print("\n--- 1. Advanced Feature Engineering ---")

# 모든 데이터 합치기 (전처리용)
all_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)

# 1. 의료 도메인 지식 기반 특성 생성
print("Creating medical domain features...")

# 갑상선 호르몬 관련 특성
all_data['TSH_T4_ratio'] = all_data['TSH_Result'] / (all_data['T4_Result'] + 1e-5)
all_data['TSH_T3_ratio'] = all_data['TSH_Result'] / (all_data['T3_Result'] + 1e-5)
all_data['T4_T3_ratio'] = all_data['T4_Result'] / (all_data['T3_Result'] + 1e-5)

# 호르몬 불균형 지표
all_data['hormone_imbalance'] = (
    np.abs(all_data['TSH_Result'] - all_data['TSH_Result'].median()) +
    np.abs(all_data['T4_Result'] - all_data['T4_Result'].median()) +
    np.abs(all_data['T3_Result'] - all_data['T3_Result'].median())
)

# 갑상선 기능 이상 지표 (TSH가 높고 T4가 낮으면 갑상선기능저하증)
all_data['hypothyroid_score'] = (
    (all_data['TSH_Result'] > all_data['TSH_Result'].quantile(0.75)).astype(int) +
    (all_data['T4_Result'] < all_data['T4_Result'].quantile(0.25)).astype(int)
)

# 갑상선 기능 항진증 지표 (TSH가 낮고 T4가 높으면)
all_data['hyperthyroid_score'] = (
    (all_data['TSH_Result'] < all_data['TSH_Result'].quantile(0.25)).astype(int) +
    (all_data['T4_Result'] > all_data['T4_Result'].quantile(0.75)).astype(int)
)

# 2. 위험 요인 조합 특성
print("Creating risk factor combinations...")

# 나이별 위험 그룹 (갑상선암은 특정 연령대에서 더 흔함)
all_data['age_risk_group'] = pd.cut(all_data['Age'], 
                                    bins=[0, 20, 40, 60, 100], 
                                    labels=['young', 'adult', 'middle', 'senior'])

# 먼저 categorical 특성들을 binary로 변환
risk_factors = ['Family_Background', 'Radiation_History', 'Iodine_Deficiency', 'Smoke', 'Diabetes']
for feature in risk_factors:
    # 'Positive'/'Yes'/1 -> 1, 'Negative'/'No'/0 -> 0
    all_data[f'{feature}_binary'] = (
        (all_data[feature] == 'Positive') | 
        (all_data[feature] == 'Yes') |
        (all_data[feature] == 1)
    ).astype(int)

# 복합 위험 점수 (이진 변수들 사용)
all_data['risk_score'] = (
    all_data['Family_Background_binary'] * 3 +  # 가족력 가중치 높게
    all_data['Radiation_History_binary'] * 2 +  # 방사선 노출도 중요
    all_data['Iodine_Deficiency_binary'] * 1.5 +
    all_data['Smoke_binary'] * 1 +
    all_data['Diabetes_binary'] * 0.5
)

# 성별-나이 상호작용 (여성의 경우 특정 연령대에서 위험 증가)
all_data['gender_age_interaction'] = (
    (all_data['Gender'] == 'Female').astype(int) * 
    ((all_data['Age'] >= 30) & (all_data['Age'] <= 50)).astype(int)
)

# 3. 결절 크기 관련 특성
print("Creating nodule-related features...")

# 결절 크기 카테고리
all_data['nodule_category'] = pd.cut(all_data['Nodule_Size'], 
                                     bins=[0, 10, 20, 30, 100],
                                     labels=['tiny', 'small', 'medium', 'large'])

# 결절 크기와 나이의 상호작용
all_data['nodule_age_ratio'] = all_data['Nodule_Size'] / (all_data['Age'] + 1)

# 결절 크기와 호르몬의 관계
all_data['nodule_tsh_interaction'] = all_data['Nodule_Size'] * all_data['TSH_Result']
all_data['nodule_hormone_score'] = (
    all_data['Nodule_Size'] * all_data['hormone_imbalance']
)

# 4. 인종별 특성 (특정 인종에서 발병률이 다를 수 있음)
print("Creating ethnicity-based features...")

# 인종별 평균 호르몬 수치와의 차이
for hormone in ['TSH_Result', 'T4_Result', 'T3_Result']:
    race_means = all_data.groupby('Race')[hormone].transform('mean')
    all_data[f'{hormone}_race_deviation'] = all_data[hormone] - race_means

# 5. 국가별 특성 (환경적 요인)
print("Creating country-based features...")

# 국가별 요오드 결핍 비율 (binary 변수 사용)
country_iodine_deficiency = all_data.groupby('Country')['Iodine_Deficiency_binary'].transform('mean')
all_data['country_iodine_risk'] = country_iodine_deficiency

# 6. 체중 위험도와 다른 요인들의 상호작용
all_data['weight_diabetes_interaction'] = (
    (all_data['Weight_Risk'] == 'High').astype(int) * 
    all_data['Diabetes_binary']
)

# 7. 포괄적인 건강 점수
all_data['overall_health_score'] = (
    (all_data['Weight_Risk'] == 'Low').astype(int) * 2 +
    (all_data['Weight_Risk'] == 'Medium').astype(int) * 1 +
    (all_data['Diabetes_binary'] == 0).astype(int) * 1 +
    (all_data['Smoke_binary'] == 0).astype(int) * 1
)

print("\n--- 2. Categorical Encoding ---")

# Label Encoding for binary features
binary_features = ['Gender', 'Family_Background', 'Radiation_History', 
                   'Iodine_Deficiency', 'Smoke', 'Diabetes']

label_encoders = {}
for feature in binary_features:
    le = LabelEncoder()
    all_data[feature] = le.fit_transform(all_data[feature])
    label_encoders[feature] = le

# One-hot encoding for multi-class features
categorical_features = ['Country', 'Race', 'Weight_Risk', 'age_risk_group', 'nodule_category']

# One-hot encoding
all_data_encoded = pd.get_dummies(all_data, columns=categorical_features, prefix=categorical_features)

print(f"Features after encoding: {all_data_encoded.shape[1]}")

print("\n--- 3. Feature Scaling ---")

# 수치형 특성 스케일링
numerical_features = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result',
                     'TSH_T4_ratio', 'TSH_T3_ratio', 'T4_T3_ratio', 'hormone_imbalance',
                     'risk_score', 'nodule_age_ratio', 'nodule_tsh_interaction',
                     'nodule_hormone_score', 'TSH_Result_race_deviation',
                     'T4_Result_race_deviation', 'T3_Result_race_deviation',
                     'country_iodine_risk', 'overall_health_score']

scaler = StandardScaler()
all_data_encoded[numerical_features] = scaler.fit_transform(all_data_encoded[numerical_features])

print("\n--- 4. Splitting and Saving ---")

# Train/Test 분리
X_train_processed = all_data_encoded[:len(X_train)]
X_test_processed = all_data_encoded[len(X_train):]

print(f"Final train shape: {X_train_processed.shape}")
print(f"Final test shape: {X_test_processed.shape}")

# 데이터 저장
X_train_processed.to_csv(RESULT_PATH + 'X_train_enhanced.csv', index=False)
X_test_processed.to_csv(RESULT_PATH + 'X_test_enhanced.csv', index=False)
y_train.to_csv(RESULT_PATH + 'y_train_enhanced.csv', index=False)
test_ids.to_csv(RESULT_PATH + 'test_ids_enhanced.csv', index=False)

print("\n--- Feature Engineering Summary ---")
print(f"Original features: {X_train.shape[1]}")
print(f"Enhanced features: {X_train_processed.shape[1]}")
print(f"New features created: {X_train_processed.shape[1] - X_train.shape[1]}")

# 특성 중요도 힌트 출력
print("\n--- Key Medical Domain Features Created ---")
print("1. Hormone ratios and imbalance indicators")
print("2. Thyroid dysfunction scores (hypo/hyperthyroid)")
print("3. Risk factor combinations and interactions")
print("4. Age-gender-nodule interactions")
print("5. Ethnicity and country-based deviations")
print("6. Comprehensive health scores")

print("\n✅ Enhanced preprocessing completed!")
