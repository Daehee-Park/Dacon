import pandas as pd
import numpy as np
import os
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Crippen
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 출력 디렉토리 생성
os.makedirs('output/try1', exist_ok=True)

CFG = {
    'NBITS': 2048,
    'SEED': 42,
    'N_FOLDS': 5,
    'N_ITER': 100,  # RandomizedSearchCV iterations
    'MORGAN_RADIUS': 2,
    'REMOVE_OUTLIERS': True,
    'USE_MOLECULAR_FEATURES': True,
    'SCALE_FEATURES': True
}

def seed_everything(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED'])

# 도메인 전문가 전처리 함수들
def calculate_molecular_descriptors(smiles):
    """도메인 지식 기반 분자 특성 계산"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptors = {
        # 기본 물리화학적 특성
        'MW': Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'Rotatable': Descriptors.NumRotatableBonds(mol),
        'Aromatic_Rings': Descriptors.NumAromaticRings(mol),
        'Heavy_Atoms': Descriptors.HeavyAtomCount(mol),
        
        # Lipinski's Rule of Five 관련
        'Lipinski_Violations': Descriptors.NumHDonors(mol) > 5 or 
                              Descriptors.NumHAcceptors(mol) > 10 or 
                              Descriptors.MolWt(mol) > 500 or 
                              Crippen.MolLogP(mol) > 5,
        
        # 추가 약물성 관련 특성
        'SlogP': Descriptors.SlogP_VSA1(mol),
        'SMR': Descriptors.SMR_VSA1(mol),
        'LabuteASA': Descriptors.LabuteASA(mol),
        'BalabanJ': Descriptors.BalabanJ(mol),
        'BertzCT': Descriptors.BertzCT(mol),
        
        # 분자 복잡도
        'FractionCsp3': Descriptors.FractionCSP3(mol),
        'RingCount': Descriptors.RingCount(mol),
        'MolLogP': Crippen.MolLogP(mol),
        'MolMR': Crippen.MolMR(mol),
    }
    
    return descriptors

def smiles_to_fingerprint(smiles, radius=2, nbits=2048):
    """개선된 Morgan Fingerprint 생성"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
        fp = morgan_gen.GetFingerprintAsNumPy(mol)
        return fp
    else:
        return np.zeros((nbits,))

def IC50_to_pIC50(ic50_nM):
    """IC50을 pIC50으로 변환 (도메인 지식 적용)"""
    # 극값 처리: 최소값을 1e-3 (매우 강한 억제제)로 설정
    ic50_nM = np.clip(ic50_nM, 1e-3, None)
    return 9 - np.log10(ic50_nM)

def pIC50_to_IC50(pIC50):
    return 10 ** (9 - pIC50)

def detect_outliers_iqr(data, column, factor=1.5):
    """IQR 기반 이상치 탐지"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def create_stratified_folds(y, molecular_features, n_splits=5):
    """분자 특성을 고려한 Stratified K-Fold"""
    # pIC50 기반 구간 생성 (도메인 지식 적용)
    pIC50_bins = pd.cut(y, bins=[0, 5, 7, 9, 15], labels=['weak', 'moderate', 'strong', 'very_strong'])
    
    # 분자량 기반 구간 생성 (Test 데이터 특성 고려)
    mw_bins = pd.cut(molecular_features['MW'], bins=4, labels=['small', 'medium', 'large', 'very_large'])
    
    # 복합 구간 생성
    combined_strata = pIC50_bins.astype(str) + '_' + mw_bins.astype(str)
    
    # StratifiedKFold 적용
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG['SEED'])
    return skf.split(molecular_features, combined_strata)

def calculate_metrics(y_true_ic50, y_pred_ic50, y_true_pic50, y_pred_pic50):
    """대회 평가 지표 계산"""
    # Normalized RMSE
    rmse = np.sqrt(mean_squared_error(y_true_ic50, y_pred_ic50))
    y_range = np.max(y_true_ic50) - np.min(y_true_ic50)
    normalized_rmse = rmse / y_range
    
    # Correlation²
    corr, _ = pearsonr(y_true_pic50, y_pred_pic50)
    correlation_squared = corr ** 2
    
    # Final Score
    A = normalized_rmse
    B = correlation_squared
    final_score = 0.4 * (1 - min(A, 1)) + 0.6 * B
    
    return normalized_rmse, correlation_squared, final_score

print("=" * 60)
print("Try #1: 도메인 지식 기반 모델 개선")
print("=" * 60)

# 1. 데이터 로드 및 전처리
print("\n1. 데이터 로드 및 도메인 전문가 전처리")
print("-" * 50)

# 원본 데이터 로드
chembl = pd.read_csv("data/ChEMBL_ASK1(IC50).csv", sep=';')
pubchem = pd.read_csv("data/Pubchem_ASK1.csv", low_memory=False)

# ChEMBL 전처리
chembl.columns = chembl.columns.str.strip().str.replace('"', '')
chembl = chembl[chembl['Standard Type'] == 'IC50']
chembl = chembl[['Smiles', 'Standard Value']].rename(columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'}).dropna()
chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')
chembl = chembl.dropna()
chembl = chembl[chembl['ic50_nM'] > 0]
chembl['source'] = 'ChEMBL'

# PubChem 전처리
pubchem = pubchem[['SMILES', 'Activity_Value']].rename(columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'}).dropna()
pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')
pubchem = pubchem.dropna()
pubchem = pubchem[pubchem['ic50_nM'] > 0]
pubchem['source'] = 'PubChem'

# 데이터 병합
total = pd.concat([chembl, pubchem], ignore_index=True)
total = total.drop_duplicates(subset='smiles')

print(f"전처리 후 데이터: {len(total):,} 개")

# 2. 분자 특성 계산 (도메인 지식 적용)
print("\n2. 분자 특성 계산")
print("-" * 50)

molecular_descriptors = []
valid_indices = []

for idx, smiles in enumerate(total['smiles']):
    descriptors = calculate_molecular_descriptors(smiles)
    if descriptors is not None:
        molecular_descriptors.append(descriptors)
        valid_indices.append(idx)

# 유효한 분자만 유지
total = total.iloc[valid_indices].reset_index(drop=True)
molecular_df = pd.DataFrame(molecular_descriptors)

print(f"분자 특성 계산 완료: {len(molecular_df)} 개 분자")
print(f"계산된 특성: {list(molecular_df.columns)}")

# 3. pIC50 계산 및 이상치 처리
total['pIC50'] = IC50_to_pIC50(total['ic50_nM'])

if CFG['REMOVE_OUTLIERS']:
    # 도메인 지식 기반 이상치 제거
    outliers_pic50 = detect_outliers_iqr(total, 'pIC50', factor=2.0)  # 관대한 기준
    outliers_mw = detect_outliers_iqr(molecular_df, 'MW', factor=2.0)
    
    total_outliers = outliers_pic50 | outliers_mw
    print(f"이상치 제거: {total_outliers.sum()} 개 ({total_outliers.mean()*100:.1f}%)")
    
    total = total[~total_outliers].reset_index(drop=True)
    molecular_df = molecular_df[~total_outliers].reset_index(drop=True)

# 4. Feature Engineering
print("\n3. Feature Engineering")
print("-" * 50)

# Morgan Fingerprint
print("Morgan Fingerprint 생성...")
fingerprints = []
for smiles in total['smiles']:
    fp = smiles_to_fingerprint(smiles, radius=CFG['MORGAN_RADIUS'], nbits=CFG['NBITS'])
    fingerprints.append(fp)

X_fingerprint = np.array(fingerprints)
print(f"Fingerprint 차원: {X_fingerprint.shape}")

# 분자 특성 결합
if CFG['USE_MOLECULAR_FEATURES']:
    # 분자 특성 정규화
    if CFG['SCALE_FEATURES']:
        scaler = RobustScaler()  # 이상치에 강건한 스케일러
        molecular_scaled = scaler.fit_transform(molecular_df)
    else:
        molecular_scaled = molecular_df.values
    
    # Fingerprint와 분자 특성 결합
    X_combined = np.concatenate([X_fingerprint, molecular_scaled], axis=1)
    feature_names = [f'morgan_{i}' for i in range(CFG['NBITS'])] + list(molecular_df.columns)
    print(f"결합된 Feature 차원: {X_combined.shape}")
else:
    X_combined = X_fingerprint
    feature_names = [f'morgan_{i}' for i in range(CFG['NBITS'])]

# 5. 도메인 지식 기반 Cross-Validation
print("\n4. 도메인 지식 기반 Cross-Validation")
print("-" * 50)

y_pic50 = total['pIC50'].values
y_ic50 = total['ic50_nM'].values

# Stratified K-Fold 생성
fold_generator = create_stratified_folds(y_pic50, molecular_df, CFG['N_FOLDS'])

cv_results = []
fold_predictions = []

print("Cross-Validation 시작...")
for fold, (train_idx, val_idx) in enumerate(fold_generator):
    print(f"\nFold {fold + 1}/{CFG['N_FOLDS']}")
    
    # 데이터 분할
    X_train, X_val = X_combined[train_idx], X_combined[val_idx]
    y_train_pic50, y_val_pic50 = y_pic50[train_idx], y_pic50[val_idx]
    y_train_ic50, y_val_ic50 = y_ic50[train_idx], y_ic50[val_idx]
    
    # 데이터소스 분포 확인
    train_sources = total.iloc[train_idx]['source'].value_counts()
    val_sources = total.iloc[val_idx]['source'].value_counts()
    print(f"  Train 데이터소스 분포: {dict(train_sources)}")
    print(f"  Val 데이터소스 분포: {dict(val_sources)}")
    
    # 하이퍼파라미터 튜닝
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.8],
        'bootstrap': [True, False]
    }
    
    rf_base = RandomForestRegressor(random_state=CFG['SEED'], n_jobs=-1)
    rf_search = RandomizedSearchCV(
        rf_base, param_dist, n_iter=CFG['N_ITER'], 
        cv=3, scoring='neg_mean_squared_error',
        random_state=CFG['SEED'], n_jobs=-1
    )
    
    rf_search.fit(X_train, y_train_pic50)
    best_model = rf_search.best_estimator_
    
    print(f"  최적 하이퍼파라미터: {rf_search.best_params_}")
    
    # 예측
    y_val_pred_pic50 = best_model.predict(X_val)
    y_val_pred_ic50 = pIC50_to_IC50(y_val_pred_pic50)
    
    # 평가 지표 계산
    normalized_rmse, correlation_squared, final_score = calculate_metrics(
        y_val_ic50, y_val_pred_ic50, y_val_pic50, y_val_pred_pic50
    )
    
    cv_results.append({
        'fold': fold + 1,
        'normalized_rmse': normalized_rmse,
        'correlation_squared': correlation_squared,
        'final_score': final_score,
        'best_params': rf_search.best_params_
    })
    
    fold_predictions.append({
        'fold': fold + 1,
        'indices': val_idx,
        'true_pic50': y_val_pic50,
        'pred_pic50': y_val_pred_pic50,
        'true_ic50': y_val_ic50,
        'pred_ic50': y_val_pred_ic50
    })
    
    print(f"  Normalized RMSE: {normalized_rmse:.4f}")
    print(f"  Correlation²: {correlation_squared:.4f}")
    print(f"  Final Score: {final_score:.4f}")

# 6. 결과 분석 및 저장
print("\n" + "=" * 60)
print("Cross-Validation 결과 분석")
print("=" * 60)

cv_df = pd.DataFrame(cv_results)
print(f"Average Normalized RMSE: {cv_df['normalized_rmse'].mean():.4f} ± {cv_df['normalized_rmse'].std():.4f}")
print(f"Average Correlation²: {cv_df['correlation_squared'].mean():.4f} ± {cv_df['correlation_squared'].std():.4f}")
print(f"Average Final Score: {cv_df['final_score'].mean():.4f} ± {cv_df['final_score'].std():.4f}")

# Baseline과 비교
baseline_score = 0.4431
improvement = cv_df['final_score'].mean() - baseline_score
print(f"\nBaseline 대비 개선: {improvement:+.4f} ({improvement/baseline_score*100:+.1f}%)")

# 결과 저장
cv_df.to_csv('output/try1/cv_results.csv', index=False)

# 예측값 저장
all_predictions = pd.DataFrame()
for pred in fold_predictions:
    fold_df = pd.DataFrame({
        'fold': pred['fold'],
        'index': pred['indices'],
        'true_pic50': pred['true_pic50'],
        'pred_pic50': pred['pred_pic50'],
        'true_ic50': pred['true_ic50'],
        'pred_ic50': pred['pred_ic50']
    })
    all_predictions = pd.concat([all_predictions, fold_df], ignore_index=True)

all_predictions.to_csv('output/try1/fold_predictions.csv', index=False)

# 7. 최종 모델 훈련 및 제출 파일 생성
print("\n5. 최종 모델 훈련")
print("-" * 50)

# 최적 하이퍼파라미터 평균 계산
best_params_avg = {}
for param in cv_results[0]['best_params'].keys():
    if param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
        values = [result['best_params'][param] for result in cv_results if result['best_params'][param] is not None]
        if values:
            best_params_avg[param] = int(np.mean(values))
    else:
        # 최빈값 사용
        values = [result['best_params'][param] for result in cv_results]
        best_params_avg[param] = max(set(values), key=values.count)

print(f"최종 하이퍼파라미터: {best_params_avg}")

# 전체 데이터로 최종 모델 훈련
final_model = RandomForestRegressor(**best_params_avg, random_state=CFG['SEED'], n_jobs=-1)
final_model.fit(X_combined, y_pic50)

# 테스트 데이터 예측
test_df = pd.read_csv("data/test.csv")

# 테스트 데이터 특성 계산
test_molecular_descriptors = []
test_fingerprints = []
valid_test_indices = []

for idx, smiles in enumerate(test_df['smiles']):
    descriptors = calculate_molecular_descriptors(smiles)
    fp = smiles_to_fingerprint(smiles, radius=CFG['MORGAN_RADIUS'], nbits=CFG['NBITS'])
    
    if descriptors is not None:
        test_molecular_descriptors.append(descriptors)
        test_fingerprints.append(fp)
        valid_test_indices.append(idx)

test_df_valid = test_df.iloc[valid_test_indices].reset_index(drop=True)
test_molecular_df = pd.DataFrame(test_molecular_descriptors)
test_fingerprint_array = np.array(test_fingerprints)

# 테스트 특성 결합
if CFG['USE_MOLECULAR_FEATURES']:
    test_molecular_scaled = scaler.transform(test_molecular_df)
    X_test_combined = np.concatenate([test_fingerprint_array, test_molecular_scaled], axis=1)
else:
    X_test_combined = test_fingerprint_array

# 예측
test_pred_pic50 = final_model.predict(X_test_combined)
test_pred_ic50 = pIC50_to_IC50(test_pred_pic50)

# 제출 파일 생성
submission = pd.DataFrame({
    'ID': test_df_valid['ID'],
    'ASK1_IC50_nM': test_pred_ic50
})

submission.to_csv('output/try1/submission.csv', index=False)

# 8. 시각화 및 분석
print("\n6. 결과 시각화")
print("-" * 50)

# CV 결과 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Fold별 성능
axes[0,0].bar(cv_df['fold'], cv_df['final_score'])
axes[0,0].set_xlabel('Fold')
axes[0,0].set_ylabel('Final Score')
axes[0,0].set_title('CV Final Score by Fold')
axes[0,0].grid(True, alpha=0.3)

# 예측 vs 실제 (pIC50)
axes[0,1].scatter(all_predictions['true_pic50'], all_predictions['pred_pic50'], alpha=0.6)
axes[0,1].plot([all_predictions['true_pic50'].min(), all_predictions['true_pic50'].max()], 
               [all_predictions['true_pic50'].min(), all_predictions['true_pic50'].max()], 'r--')
axes[0,1].set_xlabel('True pIC50')
axes[0,1].set_ylabel('Predicted pIC50')
axes[0,1].set_title('True vs Predicted pIC50')
axes[0,1].grid(True, alpha=0.3)

# 잔차 분석
residuals = all_predictions['true_pic50'] - all_predictions['pred_pic50']
axes[1,0].scatter(all_predictions['pred_pic50'], residuals, alpha=0.6)
axes[1,0].axhline(y=0, color='r', linestyle='--')
axes[1,0].set_xlabel('Predicted pIC50')
axes[1,0].set_ylabel('Residuals')
axes[1,0].set_title('Residual Plot')
axes[1,0].grid(True, alpha=0.3)

# Feature Importance (상위 20개)
feature_importance = final_model.feature_importances_
top_indices = np.argsort(feature_importance)[-20:]
top_features = [feature_names[i] for i in top_indices]
top_importance = feature_importance[top_indices]

axes[1,1].barh(range(len(top_features)), top_importance)
axes[1,1].set_yticks(range(len(top_features)))
axes[1,1].set_yticklabels(top_features, fontsize=8)
axes[1,1].set_xlabel('Feature Importance')
axes[1,1].set_title('Top 20 Feature Importance')

plt.tight_layout()
plt.savefig('output/try1/analysis_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# 설정 저장
config_df = pd.DataFrame([CFG])
config_df.to_csv('output/try1/config.csv', index=False)

print(f"\n결과 저장 완료:")
print(f"- CV 결과: output/try1/cv_results.csv")
print(f"- 예측값: output/try1/fold_predictions.csv")
print(f"- 제출파일: output/try1/submission.csv")
print(f"- 시각화: output/try1/analysis_plots.png")
print(f"- 설정: output/try1/config.csv")

print(f"\n최종 CV Score: {cv_df['final_score'].mean():.4f} ± {cv_df['final_score'].std():.4f}") 