# ------------------------- 1. Imports ---------------------------------------
import os, json, warnings
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED

from sklearn.preprocessing import StandardScaler

# ------------------------- 2. I/O -------------------------------------------
train_df = pd.read_csv('./data/train.csv')
test_df  = pd.read_csv('./data/test.csv')

# ------------------------- 3. Helper functions ------------------------------
# 핵심 분자 특성만 선별 (신약개발에 중요한 특성들)
CORE_DESCRIPTOR_FUNCS = {
    # 기본 물리화학적 특성
    'MolWt'            : Descriptors.MolWt,
    'MolLogP'          : Descriptors.MolLogP,
    'TPSA'             : rdMolDescriptors.CalcTPSA,
    
    # Lipinski Rule of Five 관련
    'NumHBD'           : rdMolDescriptors.CalcNumLipinskiHBD,
    'NumHBA'           : rdMolDescriptors.CalcNumLipinskiHBA,
    'NumRotatableBonds': Descriptors.NumRotatableBonds,
    
    # 구조적 특성
    'NumRings'         : Descriptors.RingCount,
    'NumAromaticRings' : rdMolDescriptors.CalcNumAromaticRings,
    'HeavyAtomCount'   : rdMolDescriptors.CalcNumHeavyAtoms,
    'FractionCSP3'     : rdMolDescriptors.CalcFractionCSP3,
    
    # 추가 약물성 관련 특성
    'BertzCT'          : Descriptors.BertzCT,  # 분자 복잡도
    'QED'              : QED.qed,              # QED 점수
}

def mol_from_smiles(smi: str):
    """Convert SMILES → RDKit Mol, returns None if invalid."""
    mol = Chem.MolFromSmiles(smi)
    return mol

def calc_descriptors(mol):
    """핵심 분자 특성 계산"""
    if mol is None:
        return [np.nan] * len(CORE_DESCRIPTOR_FUNCS)
    
    results = []
    for name, func in CORE_DESCRIPTOR_FUNCS.items():
        try:
            value = func(mol)
            results.append(value)
        except:
            results.append(np.nan)
    
    return results

# ------------------------- 4. Feature engineering ---------------------------
def featurize_dataframe(df: pd.DataFrame, smiles_col='Canonical_Smiles'):
    """간결한 분자 특성 추출"""
    mols = df[smiles_col].apply(mol_from_smiles)
    
    # 핵심 특성 계산
    desc_mat = np.vstack(mols.apply(calc_descriptors).values)
    desc_cols = list(CORE_DESCRIPTOR_FUNCS.keys())
    desc_df = pd.DataFrame(desc_mat, columns=desc_cols, index=df.index)
    
    return desc_df

print('간결한 분자 특성 생성 중...')
X_train_raw = featurize_dataframe(train_df)
X_test_raw  = featurize_dataframe(test_df)
y_train = train_df['Inhibition']

print(f'생성된 특성 수: {X_train_raw.shape[1]}')

# ------------------------- 5. NaN 처리 ------------------------------------
nan_rows = X_train_raw.isna().any(axis=1).sum()
if nan_rows > 0:
    warnings.warn(f'{nan_rows}개 분자에서 특성 계산 실패; 제거됩니다')
    keep_idx = ~X_train_raw.isna().any(axis=1)
    X_train_raw = X_train_raw.loc[keep_idx]
    y_train     = y_train.loc[keep_idx]

# 테스트 세트 NaN 처리
if X_test_raw.isna().any().any():
    imputation_values = X_train_raw.median()
    X_test_raw.fillna(imputation_values, inplace=True)
    X_test_raw.fillna(0, inplace=True)

# ------------------------- 6. 스케일링 ------------------------------------
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train_raw),
    columns=X_train_raw.columns,
    index=X_train_raw.index
)

X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_raw),
    columns=X_test_raw.columns,
    index=X_test_raw.index
)

# ------------------------- 7. 최종 데이터 준비 -----------------------------
X_train = X_train_scaled.copy()
X_test = X_test_scaled.copy()

# AutoGluon용 타겟 변수 추가
X_train['Inhibition'] = y_train.values

print(f'최종 특성 행렬: {X_train.shape[1]-1}개 특성; '
      f'{X_train.shape[0]}개 훈련 샘플 / {X_test.shape[0]}개 테스트 샘플')

# ------------------------- 8. 저장 ----------------------------------------
os.makedirs('./output/try10', exist_ok=True)
X_train.to_csv('./output/try10/preprocessed_train.csv', index=False)
X_test.to_csv('./output/try10/preprocessed_test.csv', index=False)

# 특성 정보 저장
feature_info = {
    'feature_count': len(X_train.columns) - 1,
    'feature_names': [col for col in X_train.columns if col != 'Inhibition'],
    'scaling_method': 'StandardScaler',
    'fingerprints_used': False,
    'description': '신약개발에 중요한 핵심 분자 특성만 사용'
}

with open('./output/try10/feature_selection_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2, ensure_ascii=False)

print("전처리 완료. './output/try10/' 폴더에 저장됨") 