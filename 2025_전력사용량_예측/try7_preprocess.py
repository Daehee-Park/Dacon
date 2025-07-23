import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import lightgbm as lgb
import logging
import os

# 로깅 설정
os.makedirs('output/try7', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/try7/preprocess.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
logger.info("==========Preprocessing Start (Try 7)==========")

# 데이터 로드
try:
    building_info = pd.read_csv('./data/building_info.csv')
    train_df = pd.read_csv('./data/train.csv')
    test_df = pd.read_csv('./data/test.csv')
    logger.info("Data loaded successfully.")
except FileNotFoundError as e:
    logger.error(f"Error loading data: {e}")
    exit()


# 컬럼명 영어로 변경
column_mapping = {
    '건물번호': 'building_id',
    '일시': 'datetime',
    '기온(°C)': 'temperature',
    '강수량(mm)': 'rainfall',
    '풍속(m/s)': 'wind_speed',
    '습도(%)': 'humidity',
    '일조(hr)': 'sunshine',
    '일사(MJ/m2)': 'solar_radiation',
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
logger.info("Renamed columns to English.")

# 데이터 병합
train_df = train_df.merge(building_info, on='building_id', how='left')
test_df = test_df.merge(building_info, on='building_id', how='left')
logger.info("Merged building info with train/test data.")

# 결측치 처리
for col in ['solar_capacity', 'ess_capacity', 'pcs_capacity']:
    train_df[col] = train_df[col].replace('-', 0).astype(float)
    test_df[col] = test_df[col].replace('-', 0).astype(float)
logger.info("Handled missing values in solar/ESS/PCS columns.")

# 시계열 특성 생성
for df in [train_df, test_df]:
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day

# --- 테스트셋에 없는 'sunshine', 'solar_radiation' 예측 모델 ---
def predict_missing_features(train, test):
    logger.info("Predicting missing features (sunshine, solar_radiation) for the test set.")
    
    # 예측할 피처와 사용할 피처 정의
    features_to_predict = ['sunshine', 'solar_radiation']
    base_features = ['temperature', 'rainfall', 'wind_speed', 'humidity', 'hour', 'day_of_week', 'month']
    
    # 테스트셋에 없는 피처 초기화
    for feat in features_to_predict:
        test[feat] = np.nan

    for feature in features_to_predict:
        logger.info(f"Training model to predict '{feature}'...")
        
        # 모델 학습 데이터 준비
        X_train_feat = train[base_features]
        y_train_feat = train[feature]
        X_test_feat = test[base_features]
        
        # LightGBM 모델
        lgbm = lgb.LGBMRegressor(random_state=42)
        lgbm.fit(X_train_feat, y_train_feat)
        
        # 테스트셋 예측 및 결측치 채우기
        test[feature] = lgbm.predict(X_test_feat)
        logger.info(f"'{feature}' prediction complete.")
        
    return train, test

train_df, test_df = predict_missing_features(train_df, test_df)
logger.info("Finished predicting missing features for the test set.")


# --- 피처 엔지니어링 ---
logger.info("Starting feature engineering.")
for df in [train_df, test_df]:
    # 주기성 피처
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

    # 휴일 피처 (간단한 예시: 8월 15일)
    df['is_holiday'] = ((df['month'] == 8) & (df['day'] == 15)).astype(int)
    df['is_weekend_holiday'] = ((df['day_of_week'] >= 5) | (df['is_holiday'] == 1)).astype(int)

    # 불쾌지수 (DI)
    df['di'] = df['temperature'] - 0.55 * (1 - 0.01 * df['humidity']) * (df['temperature'] - 14.5)
    
    # 시차 피처 (건물별로 계산)
    df.sort_values(by=['building_id', 'datetime'], inplace=True)
    df['temp_lag_1'] = df.groupby('building_id')['temperature'].shift(1)

# 시차 피처 생성 후 생긴 결측치 처리 (평균값으로)
train_df['temp_lag_1'].fillna(train_df['temperature'].mean(), inplace=True)
test_df['temp_lag_1'].fillna(test_df['temperature'].mean(), inplace=True)

logger.info("Added cyclical, holiday, DI, and lag features.")

# 건물별 시간당 평균 전력소비량
building_hour_avg = train_df.groupby(['building_id', 'hour'])['power_consumption'].median().reset_index()
building_hour_avg.rename(columns={'power_consumption': 'building_hour_median'}, inplace=True)

train_df = train_df.merge(building_hour_avg, on=['building_id', 'hour'], how='left')
test_df = test_df.merge(building_hour_avg, on=['building_id', 'hour'], how='left')
test_df['building_hour_median'].fillna(train_df['building_hour_median'].median(), inplace=True) # 테스트셋 결측치 처리
logger.info("Added building-hour median power consumption feature.")


# 왜도 처리
skewed_features = ['rainfall', 'wind_speed', 'total_area', 'cooling_area', 
                   'solar_capacity', 'ess_capacity', 'pcs_capacity', 'sunshine', 'solar_radiation']

for col in skewed_features:
    # 0인 값이 있을 수 있으므로 log1p 사용
    train_df[col] = np.log1p(train_df[col])
    test_df[col] = np.log1p(test_df[col])

# 음의 왜도를 가진 습도는 제곱 변환
train_df['humidity'] = train_df['humidity']**2
test_df['humidity'] = test_df['humidity']**2
logger.info("Applied log/power transformation to skewed features.")

# 수치형/범주형 컬럼 분리
num_features = [
    'temperature', 'rainfall', 'wind_speed', 'humidity', 
    'sunshine', 'solar_radiation', 'total_area', 'cooling_area', 
    'solar_capacity', 'ess_capacity', 'pcs_capacity',
    'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',
    'is_weekend_holiday', 'di', 'temp_lag_1', 'building_hour_median'
]
cat_features = ['building_type']

logger.info(f"Numerical features ({len(num_features)}): {num_features}")
logger.info(f"Categorical features ({len(cat_features)}): {cat_features}")

# 전처리 파이프라인
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ],
    remainder='passthrough'
)

# 특성/타겟 분리
X_train = train_df[num_features + cat_features]
X_test = test_df[num_features + cat_features]
y_train = train_df['power_consumption']

# 전처리 적용
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
logger.info("Applied preprocessing pipeline.")

# 전처리된 데이터 저장
train_processed_df = pd.DataFrame(X_train_processed, columns=preprocessor.get_feature_names_out())
train_processed_df['power_consumption'] = y_train.values

test_processed_df = pd.DataFrame(X_test_processed, columns=preprocessor.get_feature_names_out())

train_processed_df.to_csv('output/try7/preprocessed_train.csv', index=False)
test_processed_df.to_csv('output/try7/preprocessed_test.csv', index=False)

logger.info("Saved preprocessed data to output/try7/")
logger.info("==========Preprocessing Complete (Try 7)==========") 