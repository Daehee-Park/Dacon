import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, rdMolDescriptors
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
import json
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 설정
OUTPUT_DIR = 'output/data_analysis2'
LOG_FILE = f'{OUTPUT_DIR}/analysis_log.txt'

def setup_output_directory():
    """출력 디렉토리 생성"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_message(message, print_also=True):
    """로그 메시지 출력 및 파일 저장"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    
    if print_also:
        print(log_msg)
    
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')

def calculate_molecular_properties(smiles):
    """확장된 분자 특성 계산"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            props = {
                # 기본 특성
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                
                # 추가 특성
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'AromaticRings': Descriptors.NumAromaticRings(mol),
                'HeavyAtoms': Descriptors.HeavyAtomCount(mol),
                'SaturatedRings': Descriptors.NumSaturatedRings(mol),
                'AliphaticRings': Descriptors.NumAliphaticRings(mol),
                
                # 복잡성 지표
                'Complexity': Descriptors.BertzCT(mol),
                'Flexibility': Descriptors.NumRotatableBonds(mol) / max(Descriptors.HeavyAtomCount(mol), 1),
                'FractionCsp3': Descriptors.FractionCsp3(mol),
                'RingCount': Descriptors.RingCount(mol),
                
                # Lipinski 규칙
                'LipinskiViolations': sum([
                    Descriptors.MolWt(mol) > 500,
                    Descriptors.MolLogP(mol) > 5,
                    Descriptors.NumHDonors(mol) > 5,
                    Descriptors.NumHAcceptors(mol) > 10
                ])
            }
            return props
        else:
            return None
    except Exception as e:
        return None

def IC50_to_pIC50(ic50_nM):
    """IC50를 pIC50로 변환"""
    ic50_nM = np.clip(ic50_nM, 1e-10, None)
    return 9 - np.log10(ic50_nM)

def analyze_smiles_validity(df, name):
    """SMILES 유효성 분석"""
    log_message(f"\n=== {name} SMILES 유효성 분석 ===")
    
    total_count = len(df)
    valid_count = 0
    invalid_smiles = []
    
    smiles_col = None
    for col in ['Smiles', 'SMILES', 'smiles']:
        if col in df.columns:
            smiles_col = col
            break
    
    if smiles_col is None:
        log_message(f"SMILES 컬럼을 찾을 수 없음: {df.columns.tolist()}")
        return
    
    for idx, smiles in enumerate(df[smiles_col]):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1
            else:
                invalid_smiles.append((idx, smiles))
        except Exception as e:
            invalid_smiles.append((idx, smiles, str(e)))
    
    log_message(f"총 SMILES 수: {total_count}")
    log_message(f"유효한 SMILES: {valid_count} ({valid_count/total_count*100:.2f}%)")
    log_message(f"무효한 SMILES: {len(invalid_smiles)} ({len(invalid_smiles)/total_count*100:.2f}%)")
    
    if invalid_smiles:
        log_message(f"무효한 SMILES 예시 (최대 10개):")
        for i, item in enumerate(invalid_smiles[:10]):
            if len(item) == 2:
                idx, smiles = item
                log_message(f"  {idx}: {smiles}")
            else:
                idx, smiles, error = item
                log_message(f"  {idx}: {smiles} (Error: {error})")

def analyze_data_quality(df, name):
    """데이터 품질 분석"""
    log_message(f"\n=== {name} 데이터 품질 분석 ===")
    
    # 기본 정보
    log_message(f"데이터 크기: {df.shape}")
    log_message(f"컬럼: {df.columns.tolist()}")
    
    # 결측값 분석
    log_message(f"\n결측값 분석:")
    missing_info = df.isnull().sum()
    for col, missing_count in missing_info.items():
        if missing_count > 0:
            log_message(f"  {col}: {missing_count} ({missing_count/len(df)*100:.2f}%)")
    
    # 중복값 분석
    if 'Smiles' in df.columns or 'SMILES' in df.columns:
        smiles_col = 'Smiles' if 'Smiles' in df.columns else 'SMILES'
        duplicates = df[smiles_col].duplicated().sum()
        log_message(f"\n중복 SMILES: {duplicates} ({duplicates/len(df)*100:.2f}%)")
        
        if duplicates > 0:
            dup_examples = df[df[smiles_col].duplicated(keep=False)][smiles_col].value_counts().head(5)
            log_message(f"중복 SMILES 예시:")
            for smiles, count in dup_examples.items():
                log_message(f"  {smiles}: {count}회 중복")

def analyze_ic50_distribution(df, name):
    """IC50 분포 분석"""
    log_message(f"\n=== {name} IC50 분포 분석 ===")
    
    ic50_cols = ['ic50_nM', 'Standard Value', 'Activity_Value']
    ic50_col = None
    
    for col in ic50_cols:
        if col in df.columns:
            ic50_col = col
            break
    
    if ic50_col is None:
        log_message(f"IC50 컬럼을 찾을 수 없음")
        return
    
    # 숫자로 변환
    ic50_values = pd.to_numeric(df[ic50_col], errors='coerce')
    valid_ic50 = ic50_values.dropna()
    
    log_message(f"총 IC50 값: {len(ic50_values)}")
    log_message(f"유효한 IC50 값: {len(valid_ic50)} ({len(valid_ic50)/len(ic50_values)*100:.2f}%)")
    
    if len(valid_ic50) > 0:
        log_message(f"\nIC50 통계:")
        log_message(f"  최소값: {valid_ic50.min():.2e} nM")
        log_message(f"  최대값: {valid_ic50.max():.2e} nM")
        log_message(f"  중앙값: {valid_ic50.median():.2e} nM")
        log_message(f"  평균값: {valid_ic50.mean():.2e} nM")
        log_message(f"  표준편차: {valid_ic50.std():.2e} nM")
        
        # pIC50 변환
        pic50_values = IC50_to_pIC50(valid_ic50)
        log_message(f"\npIC50 통계:")
        log_message(f"  최소값: {pic50_values.min():.2f}")
        log_message(f"  최대값: {pic50_values.max():.2f}")
        log_message(f"  중앙값: {pic50_values.median():.2f}")
        log_message(f"  평균값: {pic50_values.mean():.2f}")
        log_message(f"  표준편차: {pic50_values.std():.2f}")
        
        # 분포 분석
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        log_message(f"\nIC50 분위수:")
        for p in percentiles:
            val = np.percentile(valid_ic50, p)
            log_message(f"  {p}%: {val:.2e} nM")
        
        # 이상값 분석
        Q1 = valid_ic50.quantile(0.25)
        Q3 = valid_ic50.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = valid_ic50[(valid_ic50 < lower_bound) | (valid_ic50 > upper_bound)]
        log_message(f"\n이상값 분석 (IQR 기준):")
        log_message(f"  하한: {lower_bound:.2e} nM")
        log_message(f"  상한: {upper_bound:.2e} nM")
        log_message(f"  이상값 수: {len(outliers)} ({len(outliers)/len(valid_ic50)*100:.2f}%)")
        
        if len(outliers) > 0:
            log_message(f"  극단 이상값 예시:")
            extreme_outliers = outliers.nlargest(5).tolist() + outliers.nsmallest(5).tolist()
            for val in extreme_outliers[:10]:
                log_message(f"    {val:.2e} nM (pIC50: {IC50_to_pIC50(val):.2f})")

def analyze_molecular_properties_distribution(df, name):
    """분자 특성 분포 분석"""
    log_message(f"\n=== {name} 분자 특성 분포 분석 ===")
    
    smiles_col = None
    for col in ['Smiles', 'SMILES', 'smiles']:
        if col in df.columns:
            smiles_col = col
            break
    
    if smiles_col is None:
        log_message(f"SMILES 컬럼을 찾을 수 없음")
        return
    
    # 분자 특성 계산
    log_message(f"분자 특성 계산 중... (총 {len(df)}개)")
    mol_props = []
    failed_count = 0
    
    for idx, smiles in enumerate(df[smiles_col]):
        if idx % 100 == 0:
            log_message(f"  진행: {idx}/{len(df)}", print_also=False)
        
        props = calculate_molecular_properties(smiles)
        if props is not None:
            mol_props.append(props)
        else:
            failed_count += 1
    
    log_message(f"성공: {len(mol_props)}, 실패: {failed_count}")
    
    if len(mol_props) == 0:
        log_message(f"분자 특성 계산 실패")
        return
    
    props_df = pd.DataFrame(mol_props)
    
    # 각 특성별 통계
    log_message(f"\n분자 특성 통계:")
    for col in props_df.columns:
        values = props_df[col].dropna()
        if len(values) > 0:
            log_message(f"\n{col}:")
            log_message(f"  개수: {len(values)}")
            log_message(f"  평균: {values.mean():.3f}")
            log_message(f"  표준편차: {values.std():.3f}")
            log_message(f"  최소값: {values.min():.3f}")
            log_message(f"  최대값: {values.max():.3f}")
            log_message(f"  중앙값: {values.median():.3f}")
            
            # 분위수
            q25, q75 = values.quantile([0.25, 0.75])
            log_message(f"  25%: {q25:.3f}, 75%: {q75:.3f}")
    
    return props_df

def compare_train_test_distributions(train_props, test_props):
    """Train-Test 분포 비교"""
    log_message(f"\n=== Train-Test 분자 특성 분포 비교 ===")
    
    if train_props is None or test_props is None:
        log_message(f"분자 특성 데이터가 없어 비교 불가")
        return
    
    common_cols = set(train_props.columns) & set(test_props.columns)
    log_message(f"비교 가능한 특성: {len(common_cols)}개")
    
    significant_differences = []
    
    for col in common_cols:
        train_vals = train_props[col].dropna()
        test_vals = test_props[col].dropna()
        
        if len(train_vals) > 0 and len(test_vals) > 0:
            # 통계적 차이 검정 (Mann-Whitney U test)
            try:
                statistic, p_value = stats.mannwhitneyu(train_vals, test_vals, alternative='two-sided')
                
                train_mean = train_vals.mean()
                test_mean = test_vals.mean()
                diff_percent = abs(train_mean - test_mean) / train_mean * 100 if train_mean != 0 else 0
                
                log_message(f"\n{col}:")
                log_message(f"  Train 평균: {train_mean:.3f} (n={len(train_vals)})")
                log_message(f"  Test 평균: {test_mean:.3f} (n={len(test_vals)})")
                log_message(f"  차이: {diff_percent:.1f}%")
                log_message(f"  p-value: {p_value:.6f}")
                
                if p_value < 0.05:
                    log_message(f"  *** 통계적으로 유의한 차이 ***")
                    significant_differences.append((col, diff_percent, p_value))
                
            except Exception as e:
                log_message(f"  {col}: 통계 검정 실패 - {str(e)}")
    
    log_message(f"\n=== 유의한 차이를 보이는 특성 요약 ===")
    significant_differences.sort(key=lambda x: x[1], reverse=True)  # 차이 크기순 정렬
    
    for col, diff_percent, p_value in significant_differences:
        log_message(f"{col}: {diff_percent:.1f}% 차이 (p={p_value:.6f})")

def detect_potential_label_noise(chembl_df, pubchem_df):
    """라벨링 노이즈 검출"""
    log_message(f"\n=== 라벨링 노이즈 검출 ===")
    
    # 동일 SMILES에 대한 서로 다른 IC50 값 검출
    chembl_clean = chembl_df.copy()
    pubchem_clean = pubchem_df.copy()
    
    # 컬럼명 통일
    if 'Smiles' in chembl_clean.columns:
        chembl_clean = chembl_clean.rename(columns={'Smiles': 'smiles'})
    if 'Standard Value' in chembl_clean.columns:
        chembl_clean = chembl_clean.rename(columns={'Standard Value': 'ic50_nM'})
    
    if 'SMILES' in pubchem_clean.columns:
        pubchem_clean = pubchem_clean.rename(columns={'SMILES': 'smiles'})
    if 'Activity_Value' in pubchem_clean.columns:
        pubchem_clean = pubchem_clean.rename(columns={'Activity_Value': 'ic50_nM'})
    
    # 숫자 변환
    chembl_clean['ic50_nM'] = pd.to_numeric(chembl_clean['ic50_nM'], errors='coerce')
    pubchem_clean['ic50_nM'] = pd.to_numeric(pubchem_clean['ic50_nM'], errors='coerce')
    
    # 유효한 데이터만 선택
    chembl_clean = chembl_clean.dropna(subset=['smiles', 'ic50_nM'])
    pubchem_clean = pubchem_clean.dropna(subset=['smiles', 'ic50_nM'])
    
    log_message(f"ChEMBL 유효 데이터: {len(chembl_clean)}")
    log_message(f"PubChem 유효 데이터: {len(pubchem_clean)}")
    
    # 1. ChEMBL 내부 중복 검사
    log_message(f"\n=== ChEMBL 내부 중복 SMILES 분석 ===")
    chembl_dups = chembl_clean.groupby('smiles')['ic50_nM'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    chembl_dups = chembl_dups[chembl_dups['count'] > 1].sort_values('count', ascending=False)
    
    log_message(f"중복 SMILES: {len(chembl_dups)}")
    
    if len(chembl_dups) > 0:
        log_message(f"중복 상위 10개:")
        for _, row in chembl_dups.head(10).iterrows():
            cv = row['std'] / row['mean'] if row['mean'] != 0 else 0
            log_message(f"  {row['smiles'][:50]}...")
            log_message(f"    개수: {row['count']}, CV: {cv:.2f}")
            log_message(f"    범위: {row['min']:.2e} - {row['max']:.2e} nM")
    
    # 2. PubChem 내부 중복 검사
    log_message(f"\n=== PubChem 내부 중복 SMILES 분석 ===")
    pubchem_dups = pubchem_clean.groupby('smiles')['ic50_nM'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    pubchem_dups = pubchem_dups[pubchem_dups['count'] > 1].sort_values('count', ascending=False)
    
    log_message(f"중복 SMILES: {len(pubchem_dups)}")
    
    if len(pubchem_dups) > 0:
        log_message(f"중복 상위 10개:")
        for _, row in pubchem_dups.head(10).iterrows():
            cv = row['std'] / row['mean'] if row['mean'] != 0 else 0
            log_message(f"  {row['smiles'][:50]}...")
            log_message(f"    개수: {row['count']}, CV: {cv:.2f}")
            log_message(f"    범위: {row['min']:.2e} - {row['max']:.2e} nM")
    
    # 3. ChEMBL-PubChem 교집합 분석
    log_message(f"\n=== ChEMBL-PubChem 교집합 분석 ===")
    common_smiles = set(chembl_clean['smiles']) & set(pubchem_clean['smiles'])
    log_message(f"공통 SMILES: {len(common_smiles)}")
    
    if len(common_smiles) > 0:
        conflicts = []
        for smiles in common_smiles:
            chembl_vals = chembl_clean[chembl_clean['smiles'] == smiles]['ic50_nM'].values
            pubchem_vals = pubchem_clean[pubchem_clean['smiles'] == smiles]['ic50_nM'].values
            
            chembl_mean = np.mean(chembl_vals)
            pubchem_mean = np.mean(pubchem_vals)
            
            ratio = max(chembl_mean, pubchem_mean) / min(chembl_mean, pubchem_mean)
            if ratio > 2:  # 2배 이상 차이
                conflicts.append((smiles, chembl_mean, pubchem_mean, ratio))
        
        conflicts.sort(key=lambda x: x[3], reverse=True)  # 비율순 정렬
        
        log_message(f"충돌하는 SMILES (2배 이상 차이): {len(conflicts)}")
        log_message(f"충돌 비율: {len(conflicts)/len(common_smiles)*100:.1f}%")
        
        if len(conflicts) > 0:
            log_message(f"충돌 상위 10개:")
            for smiles, chembl_mean, pubchem_mean, ratio in conflicts[:10]:
                log_message(f"  {smiles[:50]}...")
                log_message(f"    ChEMBL: {chembl_mean:.2e} nM")
                log_message(f"    PubChem: {pubchem_mean:.2e} nM")
                log_message(f"    비율: {ratio:.1f}배")

def analyze_dataset_bias(total_df):
    """데이터셋 편향 분석"""
    log_message(f"\n=== 데이터셋 편향 분석 ===")
    
    if 'source' not in total_df.columns:
        log_message(f"소스 정보가 없어 편향 분석 불가")
        return
    
    source_counts = total_df['source'].value_counts()
    log_message(f"데이터소스별 개수:")
    for source, count in source_counts.items():
        log_message(f"  {source}: {count} ({count/len(total_df)*100:.1f}%)")
    
    # pIC50 분포 차이
    if 'pIC50' in total_df.columns:
        log_message(f"\n데이터소스별 pIC50 분포:")
        for source in source_counts.index:
            source_data = total_df[total_df['source'] == source]['pIC50']
            log_message(f"{source}:")
            log_message(f"  개수: {len(source_data)}")
            log_message(f"  평균: {source_data.mean():.3f}")
            log_message(f"  표준편차: {source_data.std():.3f}")
            log_message(f"  범위: {source_data.min():.3f} - {source_data.max():.3f}")

def main():
    """메인 분석 함수"""
    
    try:
        # 초기 설정
        setup_output_directory()
        log_message("=== 데이터 분석2 시작 ===")
        log_message(f"분석 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. Train 데이터 로드
        log_message(f"\n=== Train 데이터 로드 ===")
        
        try:
            chembl = pd.read_csv("data/ChEMBL_ASK1(IC50).csv", sep=';')
            log_message(f"ChEMBL 로드 완료: {chembl.shape}")
        except Exception as e:
            log_message(f"ChEMBL 로드 실패: {str(e)}")
            return
        
        try:
            pubchem = pd.read_csv("data/Pubchem_ASK1.csv", low_memory=False)
            log_message(f"PubChem 로드 완료: {pubchem.shape}")
        except Exception as e:
            log_message(f"PubChem 로드 실패: {str(e)}")
            return
        
        # 2. Test 데이터 로드
        log_message(f"\n=== Test 데이터 로드 ===")
        
        try:
            test_df = pd.read_csv("data/test.csv")
            log_message(f"Test 로드 완료: {test_df.shape}")
            log_message(f"Test 컬럼: {test_df.columns.tolist()}")
        except Exception as e:
            log_message(f"Test 로드 실패: {str(e)}")
            return
        
        # 3. 데이터 품질 분석
        analyze_data_quality(chembl, "ChEMBL")
        analyze_data_quality(pubchem, "PubChem")
        analyze_data_quality(test_df, "Test")
        
        # 4. SMILES 유효성 분석
        analyze_smiles_validity(chembl, "ChEMBL")
        analyze_smiles_validity(pubchem, "PubChem")
        analyze_smiles_validity(test_df, "Test")
        
        # 5. IC50 분포 분석
        analyze_ic50_distribution(chembl, "ChEMBL")
        analyze_ic50_distribution(pubchem, "PubChem")
        
        # 6. 분자 특성 분포 분석
        log_message(f"\n=== 분자 특성 분포 분석 시작 ===")
        
        # ChEMBL 전처리
        chembl_clean = chembl.copy()
        chembl_clean.columns = chembl_clean.columns.str.strip().str.replace('"', '')
        chembl_clean = chembl_clean[chembl_clean['Standard Type'] == 'IC50']
        chembl_clean = chembl_clean[['Smiles', 'Standard Value']].rename(columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'}).dropna()
        chembl_clean['ic50_nM'] = pd.to_numeric(chembl_clean['ic50_nM'], errors='coerce')
        chembl_clean['pIC50'] = IC50_to_pIC50(chembl_clean['ic50_nM'])
        chembl_clean['source'] = 'ChEMBL'
        
        # PubChem 전처리
        pubchem_clean = pubchem[['SMILES', 'Activity_Value']].rename(columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'}).dropna()
        pubchem_clean['ic50_nM'] = pd.to_numeric(pubchem_clean['ic50_nM'], errors='coerce')
        pubchem_clean['pIC50'] = IC50_to_pIC50(pubchem_clean['ic50_nM'])
        pubchem_clean['source'] = 'PubChem'
        
        # 전체 train 데이터
        total_train = pd.concat([chembl_clean, pubchem_clean], ignore_index=True)
        total_train = total_train.drop_duplicates(subset='smiles')
        total_train = total_train[total_train['ic50_nM'] > 0].dropna()
        
        log_message(f"정제된 Train 데이터: {len(total_train)}")
        
        # 분자 특성 계산
        train_props = analyze_molecular_properties_distribution(total_train, "Train")
        test_props = analyze_molecular_properties_distribution(test_df, "Test")
        
        # 7. Train-Test 분포 비교
        compare_train_test_distributions(train_props, test_props)
        
        # 8. 라벨링 노이즈 검출
        detect_potential_label_noise(chembl, pubchem)
        
        # 9. 데이터셋 편향 분석
        analyze_dataset_bias(total_train)
        
        # 10. 최종 요약
        log_message(f"\n=== 분석 요약 ===")
        log_message(f"ChEMBL 데이터: {len(chembl_clean)}")
        log_message(f"PubChem 데이터: {len(pubchem_clean)}")
        log_message(f"전체 Train 데이터: {len(total_train)}")
        log_message(f"Test 데이터: {len(test_df)}")
        log_message(f"분석 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 결과를 JSON으로도 저장
        summary = {
            'analysis_time': datetime.now().isoformat(),
            'data_sizes': {
                'chembl': len(chembl_clean),
                'pubchem': len(pubchem_clean),
                'train_total': len(total_train),
                'test': len(test_df)
            },
            'train_molecular_properties': train_props.describe().to_dict() if train_props is not None else None,
            'test_molecular_properties': test_props.describe().to_dict() if test_props is not None else None
        }
        
        with open(f'{OUTPUT_DIR}/analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        log_message(f"=== 분석 완료 ===")
        
    except Exception as e:
        error_msg = f"분석 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        log_message(error_msg)

if __name__ == "__main__":
    main() 