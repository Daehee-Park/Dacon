import logging
import os

# 로그 디렉토리 생성
os.makedirs('output', exist_ok=True)

# 파일 핸들러로 모든 로그 저장
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('output/data_analysis.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
logger.info("==========Data Analysis Start==========")

import pandas as pd

building_info = pd.read_csv('./data/building_info.csv')
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

# 건물번호 기준으로 building_info 병합
train_df = train_df.merge(building_info, on='건물번호', how='left')
test_df = test_df.merge(building_info, on='건물번호', how='left')

# 결측치 처리 추가: '-' 값을 0으로 대체
for col in ['태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']:
    train_df[col] = train_df[col].replace('-', 0).astype(float)
    test_df[col] = test_df[col].replace('-', 0).astype(float)
logger.info("Handled missing values in solar/ESS/PCS columns")

logger.info(f"train_df object columns: {train_df.select_dtypes(include='object').columns.tolist()}")
logger.info(f"train_df numerical columns: {train_df.select_dtypes(include=['number']).columns.tolist()}")

logger.info(f"train_df shape: {train_df.shape}")
logger.info(f"test_df shape: {test_df.shape}")

# 컬럼 차이 분석 추가
train_cols = set(train_df.columns)
test_cols = set(test_df.columns)

# 훈련에만 있는 컬럼
train_only = train_cols - test_cols
# 테스트에만 있는 컬럼
test_only = test_cols - train_cols
# 공통 컬럼
common_cols = train_cols & test_cols

logger.info(f"Train only columns: {list(train_only)}")
logger.info(f"Test only columns: {list(test_only)}")
logger.info(f"Common columns: {len(common_cols)}")

# 날짜 범위 분석
train_df['일시'] = pd.to_datetime(train_df['일시'])
test_df['일시'] = pd.to_datetime(test_df['일시'])
logger.info(f"Train date range: {train_df['일시'].min()} to {train_df['일시'].max()}")
logger.info(f"Test date range: {test_df['일시'].min()} to {test_df['일시'].max()}")

# 시간 기반 특성 생성 추가
for df in [train_df, test_df]:
    df['hour'] = df['일시'].dt.hour
    df['day_of_week'] = df['일시'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['month'] = df['일시'].dt.month
logger.info("Created time-based features: hour, day_of_week, is_weekend, month")

# 추가: 수치형 열 통계
num_cols = train_df.select_dtypes(include=['number']).columns
logger.info(f"Numerical columns for statistics: {list(num_cols)}")
for col in num_cols:
    logger.info(f"{col} statistics: mean={train_df[col].mean():.2f}, min={train_df[col].min()}, max={train_df[col].max()}")

# 추가: 범주형 열 분포
cat_cols = train_df.select_dtypes(include='object').columns
for col in cat_cols:
    if col != '일시':  # 일시는 이미 datetime으로 변환됨
        logger.info(f"{col} value counts:\n{str(train_df[col].value_counts().head(10))}")

# 추가: 타겟 변수 분석
logger.info(f"Target(전력소비량) statistics:\n{str(train_df['전력소비량(kWh)'].describe())}")

# 왜도 분석 및 변환 제안
from scipy.stats import skew

num_cols = train_df.select_dtypes(include=['number']).columns
skew_results = {}
for col in num_cols:
    col_skew = skew(train_df[col].dropna())
    skew_results[col] = col_skew
    logger.info(f"{col} 왜도: {col_skew:.4f}")
    
    # 변환 제안
    if abs(col_skew) > 0.5:
        suggestion = "로그 변환 권장" if col_skew > 0 else "제곱 변환 권장"
        logger.info(f"  → {suggestion} (|왜도| > 0.5)")

# 상관관계 분석 (타겟과의 관계) - 수정: 숫자형 컬럼만 선택
numeric_df = train_df.select_dtypes(include=['number'])
corr_with_target = numeric_df.corr()['전력소비량(kWh)'].sort_values(ascending=False)
logger.info("타겟 변수와의 상관관계:\n" + str(corr_with_target))

logger.info("train_df 결측치:\n" + str(train_df.isnull().sum()))
logger.info("test_df 결측치:\n" + str(test_df.isnull().sum()))

# 일조/일사량 특성 중요도 분석 추가
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. 특성 중요도 분석
X = train_df.drop(columns=['전력소비량(kWh)', 'num_date_time', '일시'])
y = train_df['전력소비량(kWh)']

# 범주형 변수 인코딩 (간소화)
X_encoded = pd.get_dummies(X, columns=['건물유형'])

# 모델 학습
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_encoded, y)

# 특성 중요도
feature_importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
top_features = feature_importances.sort_values(ascending=False).head(10)
logger.info("상위 10개 특성 중요도:\n" + str(top_features))

# 일조/일사 변수 중요도 명시적 확인
for feat in ['일조(hr)', '일사(MJ/m2)']:
    if feat in feature_importances.index:
        imp = feature_importances[feat]
        rank = int(feature_importances.rank(ascending=False)[feat])
        logger.info(f"{feat} importance: {imp:.4f}, rank: {rank}")

# 퍼뮤테이션 중요도 분석
from sklearn.inspection import permutation_importance
perm_imp = permutation_importance(model, X_encoded, y, n_repeats=10, random_state=42)
for feat in ['일조(hr)', '일사(MJ/m2)']:
    if feat in X_encoded.columns:
        idx = X_encoded.columns.get_loc(feat)
        logger.info(f"Permutation importance of {feat}: {perm_imp.importances_mean[idx]:.4f}")

# 2. 일조/일사량 예측 가능성 분석
def evaluate_feature_prediction(feature_name):
    """특성 예측 정확도 평가"""
    logger.info(f"Starting evaluation for: {feature_name}")
    
    # 공통 특성 선택
    common_features = [
        '건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', 
        '건물유형', 'hour', 'month', 'is_weekend'
    ]
    
    logger.info(f"Using common features: {common_features}")
    
    # 데이터 준비
    X = train_df[common_features]
    y = train_df[feature_name]
    
    # 범주형 변수 인코딩
    logger.info("Encoding categorical features...")
    X_encoded = pd.get_dummies(X, columns=['건물유형'])
    logger.info(f"Encoded features shape: {X_encoded.shape}")
    
    # 데이터 분할
    logger.info("Splitting data into train/validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )
    logger.info(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    
    # 모델 학습 및 예측
    logger.info("Training RandomForest model...")
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    logger.info("Model training complete")
    
    preds = model.predict(X_val)
    
    # 평가 지표
    mae = mean_absolute_error(y_val, preds)
    logger.info(f"{feature_name} 예측 성능: MAE={mae:.4f}, 평균값={y_val.mean():.4f}")
    return mae

# 일조/일사량 예측 정확도 평가
sunshine_mae = evaluate_feature_prediction('일조(hr)')
solar_mae = evaluate_feature_prediction('일사(MJ/m2)')

# 예측 가능성 판단
def assess_predictability(mae, feature_name):
    """MAE 기반 예측 가능성 평가"""
    if mae < 0.1 * train_df[feature_name].mean():
        return f"{feature_name} 예측 가능성: 우수 (예측값 활용 추천, MAE={mae:.4f})"
    elif mae < 0.2 * train_df[feature_name].mean():
        return f"{feature_name} 예측 가능성: 보통 (제한적 활용 가능, MAE={mae:.4f})"
    else:
        return f"{feature_name} 예측 가능성: 미흡 (활용 비추천, MAE={mae:.4f})"

logger.info(f"일조(hr) 예측 가능성: {assess_predictability(sunshine_mae, '일조(hr)')}")
logger.info(f"일사(MJ/m2) 예측 가능성: {assess_predictability(solar_mae, '일사(MJ/m2)')}")
logger.info("==========Data Analysis End==========")