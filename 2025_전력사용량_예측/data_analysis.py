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

"""
train_df 컬럼: ['num_date_time', '건물번호', '일시', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)',
'일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)', '건물유형', '연면적(m2)', '냉방면적(m2)',
'태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
"""
logger.info("train_df ")
# object dtype columns: ['num_date_time', '일시', '건물유형', '태양광용량(kW)', 'ESS저장용량(kWh)', 'PCS용량(kW)']
# numerical dtype columns: ['건물번호', '기온(°C)', '강수량(mm)', '풍속(m/s)', '습도(%)', '일조(hr)', '일사(MJ/m2)', '전력소비량(kWh)', '연면적(m2)', '냉방면적(m2)']

logger.info(f"train_df shape: {train_df.shape}")
logger.info(f"test_df shape: {test_df.shape}")

logger.info("train_df 결측치:\n" + str(train_df.isnull().sum()))
logger.info("test_df 결측치:\n" + str(test_df.isnull().sum()))

logger.info("==========Data Analysis End==========")