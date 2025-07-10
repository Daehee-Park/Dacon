import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem import rdFingerprintGenerator
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

def calculate_molecular_properties(smiles):
    """SMILES로부터 분자 특성 계산"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            'MW': np.nan, 'LogP': np.nan, 'HBD': np.nan, 'HBA': np.nan,
            'TPSA': np.nan, 'Rotatable': np.nan, 'Aromatic': np.nan,
            'Heavy_Atoms': np.nan, 'SMILES_Length': len(smiles)
        }
    
    return {
        'MW': Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'Rotatable': Descriptors.NumRotatableBonds(mol),
        'Aromatic': Descriptors.NumAromaticRings(mol),
        'Heavy_Atoms': Descriptors.HeavyAtomCount(mol),
        'SMILES_Length': len(smiles)
    }

def IC50_to_pIC50(ic50_nM):
    ic50_nM = np.clip(ic50_nM, 1e-10, None)
    return 9 - np.log10(ic50_nM)

print("=" * 60)
print("AI 신약개발 경진대회 - 데이터 분석")
print("=" * 60)

# 1. 원본 데이터 로드 및 기본 정보
print("\n1. 데이터 로드 및 기본 정보")
print("-" * 40)

chembl = pd.read_csv("data/ChEMBL_ASK1(IC50).csv", sep=';')
pubchem = pd.read_csv("data/Pubchem_ASK1.csv", low_memory=False)
test_df = pd.read_csv("data/test.csv")

print(f"ChEMBL 원본 데이터: {len(chembl):,} 행")
print(f"PubChem 원본 데이터: {len(pubchem):,} 행")
print(f"Test 데이터: {len(test_df):,} 행")

# 2. 데이터 전처리 후 정보
print("\n2. 전처리 후 데이터")
print("-" * 40)

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

print(f"ChEMBL 전처리 후: {len(chembl):,} 행")
print(f"PubChem 전처리 후: {len(pubchem):,} 행")

# 전체 훈련 데이터
train_data = pd.concat([chembl, pubchem], ignore_index=True)
print(f"전체 훈련 데이터 (병합 전): {len(train_data):,} 행")

# 중복 제거
train_data = train_data.drop_duplicates(subset='smiles')
print(f"전체 훈련 데이터 (중복 제거 후): {len(train_data):,} 행")

# pIC50 계산
train_data['pIC50'] = IC50_to_pIC50(train_data['ic50_nM'])

# 3. IC50 및 pIC50 분포 분석
print("\n3. IC50/pIC50 분포 분석")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# IC50 분포 (로그 스케일)
axes[0,0].hist(np.log10(train_data['ic50_nM']), bins=50, alpha=0.7, edgecolor='black')
axes[0,0].set_xlabel('log10(IC50 nM)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('IC50 Distribution (log scale)')
axes[0,0].grid(True, alpha=0.3)

# pIC50 분포
axes[0,1].hist(train_data['pIC50'], bins=50, alpha=0.7, edgecolor='black')
axes[0,1].set_xlabel('pIC50')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('pIC50 Distribution')
axes[0,1].grid(True, alpha=0.3)

# 데이터 소스별 분포
for source in ['ChEMBL', 'PubChem']:
    data_subset = train_data[train_data['source'] == source]
    axes[1,0].hist(data_subset['pIC50'], bins=30, alpha=0.6, label=source, edgecolor='black')
axes[1,0].set_xlabel('pIC50')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('pIC50 Distribution by Source')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Box plot
train_data.boxplot(column='pIC50', by='source', ax=axes[1,1])
axes[1,1].set_title('pIC50 Distribution by Source (Box Plot)')
axes[1,1].set_xlabel('Source')
axes[1,1].set_ylabel('pIC50')

plt.tight_layout()
plt.savefig('output/data_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 통계 요약
print("\n데이터 소스별 통계:")
print(train_data.groupby('source')['pIC50'].describe())
print(f"\n전체 pIC50 범위: {train_data['pIC50'].min():.2f} ~ {train_data['pIC50'].max():.2f}")

# 4. 분자 특성 분석
print("\n4. 분자 특성 분석")
print("-" * 40)

# 훈련 데이터 분자 특성 계산
print("훈련 데이터 분자 특성 계산 중...")
train_props = []
for smiles in train_data['smiles']:
    train_props.append(calculate_molecular_properties(smiles))
train_props_df = pd.DataFrame(train_props)
train_props_df['dataset'] = 'Train'

# 테스트 데이터 분자 특성 계산
print("테스트 데이터 분자 특성 계산 중...")
test_props = []
for smiles in test_df['Smiles']:
    test_props.append(calculate_molecular_properties(smiles))
test_props_df = pd.DataFrame(test_props)
test_props_df['dataset'] = 'Test'

# 전체 특성 데이터
all_props = pd.concat([train_props_df, test_props_df], ignore_index=True)

# 분자 특성 비교 시각화
fig, axes = plt.subplots(3, 3, figsize=(18, 15))
properties = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'Rotatable', 'Aromatic', 'Heavy_Atoms', 'SMILES_Length']

for i, prop in enumerate(properties):
    row, col = i // 3, i % 3
    
    # Train vs Test 분포 비교
    train_vals = all_props[all_props['dataset'] == 'Train'][prop].dropna()
    test_vals = all_props[all_props['dataset'] == 'Test'][prop].dropna()
    
    axes[row, col].hist(train_vals, bins=30, alpha=0.6, label='Train', density=True, edgecolor='black')
    axes[row, col].hist(test_vals, bins=30, alpha=0.6, label='Test', density=True, edgecolor='black')
    axes[row, col].set_xlabel(prop)
    axes[row, col].set_ylabel('Density')
    axes[row, col].set_title(f'{prop} Distribution')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('output/molecular_properties.png', dpi=300, bbox_inches='tight')
plt.show()

# 분자 특성 통계 비교
print("\n분자 특성 통계 비교 (Train vs Test):")
comparison_stats = all_props.groupby('dataset')[properties].describe()
print(comparison_stats.round(2))

# 5. 데이터 품질 체크
print("\n5. 데이터 품질 체크")
print("-" * 40)

# Invalid SMILES 체크
def check_smiles_validity(df, smiles_col, name):
    invalid_count = 0
    for smiles in df[smiles_col]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            invalid_count += 1
    
    print(f"{name}: {invalid_count}/{len(df)} invalid SMILES ({invalid_count/len(df)*100:.2f}%)")
    return invalid_count

train_invalid = check_smiles_validity(train_data, 'smiles', 'Train data')
test_invalid = check_smiles_validity(test_df, 'Smiles', 'Test data')

# 6. 중복 분석
print("\n6. 중복 및 overlap 분석")
print("-" * 40)

# Train 내 중복
print(f"Train 데이터 내 SMILES 중복: {train_data['smiles'].duplicated().sum()}개")

# Train-Test overlap
train_smiles = set(train_data['smiles'])
test_smiles = set(test_df['Smiles'])
overlap = train_smiles.intersection(test_smiles)
print(f"Train-Test SMILES overlap: {len(overlap)}개 ({len(overlap)/len(test_smiles)*100:.2f}%)")

if len(overlap) > 0:
    print("Overlap SMILES 예시 (최대 5개):")
    for i, smiles in enumerate(list(overlap)[:5]):
        print(f"  {i+1}. {smiles}")

# 7. 요약 리포트
print("\n" + "=" * 60)
print("데이터 분석 요약 리포트")
print("=" * 60)
print(f"총 훈련 데이터: {len(train_data):,}개")
print(f"총 테스트 데이터: {len(test_df):,}개")
print(f"pIC50 범위: {train_data['pIC50'].min():.2f} ~ {train_data['pIC50'].max():.2f}")
print(f"CV Score: 0.4431 ± 0.0170")
print(f"Public LB Score: 0.3569")
print(f"CV-Public 갭: -0.0862 (19.4% 하락)")
print(f"Train-Test SMILES overlap: {len(overlap)/len(test_smiles)*100:.1f}%")

print("\n그래프가 저장되었습니다:")
print("- output/data_distribution.png")
print("- output/molecular_properties.png") 