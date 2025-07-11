import pandas as pd

train_path = 'data/train.csv'
test_path = 'data/test.csv'

# 데이터 로드
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

log = []

# 데이터 크기
log.append(f"Train shape: {train.shape}")
log.append(f"Test shape: {test.shape}")

# 컬럼명
log.append(f"Train columns: {list(train.columns)}")
log.append(f"Test columns: {list(test.columns)}")

# 결측치
log.append("Train missing values:\n" + str(train.isnull().sum()))
log.append("Test missing values:\n" + str(test.isnull().sum()))

# 기본 통계
log.append("Train describe:\n" + str(train.describe(include='all')))
log.append("Test describe:\n" + str(test.describe(include='all')))

# 각 컬럼별 고유값 개수
log.append("Train nunique:\n" + str(train.nunique()))
log.append("Test nunique:\n" + str(test.nunique()))

# 타겟 분포 (타겟 컬럼명이 'target' 또는 'inhibition' 등일 수 있음, 자동 탐색)
target_col = None
for col in train.columns:
    if 'inhib' in col.lower() or 'target' in col.lower():
        target_col = col
        break
if target_col:
    log.append(f"Target column: {target_col}")
    log.append(f"Target describe:\n{train[target_col].describe()}")
else:
    log.append("No obvious target column found.")

# 결과 저장
with open('output/data_analysis.txt', 'w', encoding='utf-8') as f:
    for l in log:
        f.write(l + '\n\n')

print("기본 데이터 분석 결과가 output/data_analysis.txt에 저장되었습니다.")
