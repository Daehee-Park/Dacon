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

print("=== Step 1: Original Data Basic Preprocessing ===")

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

print("\n--- Basic Preprocessing ---")

# 모든 데이터 합치기
all_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)

print("Original features:")
for i, col in enumerate(all_data.columns, 1):
    print(f"{i:2d}. {col}")

# Label Encoding for binary features
binary_features = ['Gender', 'Family_Background', 'Radiation_History', 
                   'Iodine_Deficiency', 'Smoke', 'Diabetes']

label_encoders = {}
for feature in binary_features:
    le = LabelEncoder()
    all_data[feature] = le.fit_transform(all_data[feature])
    label_encoders[feature] = le

# One-hot encoding for multi-class features
categorical_features = ['Country', 'Race', 'Weight_Risk']
all_data_encoded = pd.get_dummies(all_data, columns=categorical_features, prefix=categorical_features)

print(f"Features after encoding: {all_data_encoded.shape[1]}")

# Feature Scaling
numerical_features = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']

scaler = StandardScaler()
all_data_final = all_data_encoded.copy()
all_data_final[numerical_features] = scaler.fit_transform(all_data_encoded[numerical_features])

# Train/Test 분리
X_train_processed = all_data_final[:len(X_train)]
X_test_processed = all_data_final[len(X_train):]

print(f"Final train shape: {X_train_processed.shape}")
print(f"Final test shape: {X_test_processed.shape}")

# 데이터 저장
X_train_processed.to_csv(RESULT_PATH + 'X_train.csv', index=False)
X_test_processed.to_csv(RESULT_PATH + 'X_test.csv', index=False)
y_train.to_csv(RESULT_PATH + 'y_train.csv', index=False)
test_ids.to_csv(RESULT_PATH + 'test_ids.csv', index=False)

print(f"\nStep 1 완료!")
print(f"특성 수: {X_train_processed.shape[1]}개")
print(f"저장된 파일:")
print(f"- X_train.csv, X_test.csv, y_train.csv, test_ids.csv")

# 다음 단계 실행 명령어 출력
print(f"\n다음 단계 실행:")
print(f"python step2_optimize_lgb.py") 