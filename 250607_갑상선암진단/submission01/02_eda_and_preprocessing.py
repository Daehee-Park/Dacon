import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# 데이터 경로 설정
DATA_PATH = './data/'

# 데이터 불러오기
train_df = pd.read_csv(DATA_PATH + 'train.csv')
test_df = pd.read_csv(DATA_PATH + 'test.csv')

# ID는 나중에 submission 파일을 만들 때 필요하므로 test_ids에 저장해두고 train/test 데이터프레임에서는 삭제합니다.
train_df = train_df.drop('ID', axis=1)
test_ids = test_df['ID']
test_df = test_df.drop('ID', axis=1)

print("--- 1. EDA: Visualizing Feature Distributions ---")

# 수치형 피처와 범주형 피처 리스트 정의
numerical_features = ['Age', 'Nodule_Size', 'TSH_Result', 'T4_Result', 'T3_Result']
categorical_features = ['Gender', 'Country', 'Race', 'Family_Background', 
                        'Radiation_History', 'Iodine_Deficiency', 'Smoke', 
                        'Weight_Risk', 'Diabetes']

# 수치형 피처 분포 시각화 (KDE plot과 Box plot)
for col in numerical_features:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # KDE plot
    sns.kdeplot(data=train_df, x=col, hue='Cancer', fill=True, ax=axes[0])
    axes[0].set_title(f'KDE Plot of {col} by Cancer')
    
    # Box plot
    sns.boxplot(data=train_df, x='Cancer', y=col, ax=axes[1])
    axes[1].set_title(f'Box Plot of {col} by Cancer')
    
    plt.tight_layout()
    plt.show()

# 범주형 피처 분포 시각화 (Count plot)
for col in categorical_features:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=train_df, x=col, hue='Cancer')
    plt.title(f'Distribution of {col} by Cancer')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
# 수치형 피처간의 상관관계 히트맵
plt.figure(figsize=(10, 8))
sns.heatmap(train_df[numerical_features + ['Cancer']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()

print("\n--- 2. Data Preprocessing ---")

# 효율적인 전처리를 위해 train과 test 데이터 합치기
all_df = pd.concat([train_df.drop('Cancer', axis=1), test_df], axis=0)

# 범주형 피처 인코딩
# 이진 변수는 Label Encoding
binary_cols = [col for col in categorical_features if all_df[col].nunique() == 2]
print(f"Applying Label Encoding to: {binary_cols}")
for col in binary_cols:
    le = LabelEncoder()
    all_df[col] = le.fit_transform(all_df[col])

# 다중 클래스 변수는 One-Hot Encoding
multi_class_cols = [col for col in categorical_features if all_df[col].nunique() > 2]
print(f"Applying One-Hot Encoding to: {multi_class_cols}")
all_df = pd.get_dummies(all_df, columns=multi_class_cols, drop_first=True)

# 수치형 피처 스케일링
print(f"Applying StandardScaler to: {numerical_features}")
scaler = StandardScaler()
all_df[numerical_features] = scaler.fit_transform(all_df[numerical_features])

# 전처리된 데이터를 다시 train과 test로 분리
X = all_df[:len(train_df)].copy()
y = train_df['Cancer'].copy()
X_test = all_df[len(train_df):].copy()

print("\n--- 3. Saving Processed Data ---")
# 처리된 데이터 저장
X.to_csv(DATA_PATH + 'X_train_processed.csv', index=False)
y.to_csv(DATA_PATH + 'y_train_processed.csv', index=False)
X_test.to_csv(DATA_PATH + 'X_test_processed.csv', index=False)
test_ids.to_csv(DATA_PATH + 'test_ids.csv', index=False)

print("Processed data saved successfully.")
print(f"Train data shape after processing: {X.shape}")
print(f"Test data shape after processing: {X_test.shape}")
