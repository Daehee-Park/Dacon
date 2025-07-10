import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 데이터 경로 설정
DATA_PATH = './data/'

# 데이터 불러오기
train_df = pd.read_csv(DATA_PATH + 'train.csv')
test_df = pd.read_csv(DATA_PATH + 'test.csv')
submission_df = pd.read_csv(DATA_PATH + 'sample_submission.csv')

# --- 1. 데이터 기본 정보 확인 ---
print('--- Train Data ---')
print(train_df.head())
print(f'Train data shape: {train_df.shape}')
print('\n')

print('--- Test Data ---')
print(test_df.head())
print(f'Test data shape: {test_df.shape}')
print('\n')

# --- 2. 데이터 정보 및 결측치 확인 ---
print('--- Train Data Info & Nulls ---')
train_df.info()
print(train_df.isnull().sum())
print('\n')

print('--- Test Data Info & Nulls ---')
test_df.info()
print(test_df.isnull().sum())
print('\n')

# --- 3. 타겟 변수 분포 확인 ---
print('--- Target Distribution ---')
print(train_df['Cancer'].value_counts())
print(train_df['Cancer'].value_counts(normalize=True))

plt.figure(figsize=(8, 6))
sns.countplot(x='Cancer', data=train_df)
plt.title('Target Distribution')
plt.show()

# --- 4. 수치형/범주형 데이터 통계량 확인 ---
print("--- Numerical Feature Statistics (Train) ---")
print(train_df.describe())
print("\n")

print("--- Categorical Feature Statistics (Train) ---")
print(train_df.describe(include='object'))
print("\n")
