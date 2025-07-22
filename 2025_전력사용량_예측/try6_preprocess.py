import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import logging
import os

# 로깅 설정
os.makedirs('output', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/try6/preprocess.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
logger.info("==========Preprocessing Start==========")

# 데이터 로드
building_info = pd.read_csv('./data/building_info.csv')
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

# 컬럼명 영어로 변경
column_mapping = {
    '건물번호': 'building_id',
    '일시': 'datetime',
    '기온(°C)': 'temperature',
    '강수량(mm)': 'rainfall',
    '풍속(m/s)': 'wind_speed',
    '습도(%)': 'humidity',
    '전력소비량(kWh)': 'power_consumption',
    '건물유형': 'building_type',
    '연면적(m2)': 'total_area',
    '냉방면적(m2)': 'cooling_area',
    '태양광용량(kW)': 'solar_capacity',
    'ESS저장용량(kWh)': 'ess_capacity',
    'PCS용량(kW)': 'pcs_capacity',
    'num_date_time': 'num_datetime'
}

building_info.rename(columns=column_mapping, inplace=True)
train_df.rename(columns=column_mapping, inplace=True)
test_df.rename(columns=column_mapping, inplace=True)
logger.info("Renamed columns to English")

# 데이터 병합
train_df = train_df.merge(building_info, on='building_id', how='left')
test_df = test_df.merge(building_info, on='building_id', how='left')
logger.info("Merged building info with train/test data")

# 결측치 처리
for col in ['solar_capacity', 'ess_capacity', 'pcs_capacity']:
    train_df[col] = train_df[col].replace('-', 0).astype(float)
    test_df[col] = test_df[col].replace('-', 0).astype(float)
logger.info("Handled missing values in solar/ESS/PCS columns")

# 시계열 특성 추출
for df in [train_df, test_df]:
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['month'] = df['datetime'].dt.month
logger.info("Extracted time-based features")

# 왜도 처리
skewed_features = ['rainfall', 'wind_speed', 'total_area', 'cooling_area']
for col in skewed_features:
    train_df[col] = np.log1p(train_df[col])
    test_df[col] = np.log1p(test_df[col])
logger.info("Applied log transformation to skewed features")

# 특성 공학
building_hour_avg = train_df.groupby(['building_type', 'hour'])['power_consumption'].median().reset_index()
building_hour_avg.rename(columns={'power_consumption': 'type_hour_avg'}, inplace=True)

train_df = train_df.merge(building_hour_avg, on=['building_type', 'hour'], how='left')
test_df = test_df.merge(building_hour_avg, on=['building_type', 'hour'], how='left')

for df in [train_df, test_df]:
    df['temp_times_hour'] = df['temperature'] * df['hour']
    df['is_weekday'] = 1 - df['is_weekend']
logger.info("Applied feature engineering")

# 수치형/범주형 컬럼 분리
num_features = [
    'temperature', 'rainfall', 'wind_speed', 'humidity', 'total_area', 
    'cooling_area', 'solar_capacity', 'ess_capacity', 'pcs_capacity',
    'temp_times_hour', 'type_hour_avg', 'hour', 'day_of_week', 'month', 'is_weekend'
]
cat_features = ['building_type']

# 전처리 파이프라인
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

# 특성/타겟 분리
X_train = train_df[num_features + cat_features]
X_test = test_df[num_features + cat_features]
y_train = train_df['power_consumption']

# 전처리 적용
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
logger.info("Applied preprocessing pipeline")

# 전처리된 데이터 저장
os.makedirs('output/try6', exist_ok=True)

train_processed = pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out())
train_processed['power_consumption'] = y_train.values
train_processed.to_csv('output/try6/preprocessed_train.csv', index=False)

test_processed = pd.DataFrame(X_test_processed, columns=preprocessor.get_feature_names_out())
test_processed.to_csv('output/try6/preprocessed_test.csv', index=False)

logger.info("Saved preprocessed data to output/try6/")
logger.info("==========Preprocessing Complete==========") 