 # ------------------------- 1. Imports ---------------------------------------
import os, json, warnings
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, AllChem

from sklearn.preprocessing import StandardScaler

# ------------------------- 2. I/O -------------------------------------------
train_df = pd.read_csv('./data/train.csv')
test_df  = pd.read_csv('./data/test.csv')

# ------------------------- 3. Known CYP3A4 inhibitors ----------------------
# 논문에서 언급된 저해제들의 SMILES
KNOWN_INHIBITORS = {
    'ketoconazole': 'CC1=CC(=C(C=C1)N2C(=O)C3=C(C=CC(=C3)Cl)N(C2=O)CCN4C=CN=C4)C(C)(C)C',
    'bergapten': 'COC1=CC2=C(C=C1)C(=O)OC3=C2C=CC(=C3)OC',
    'quercetin': 'C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O',
    'piperine': 'CCN(CC)C(=O)C1=CC=C(C=C1)C2=C3C=CC(=CC3=C(C=C2)C(F)(F)F)Cl'
}

# ------------------------- 4. Helper functions ------------------------------
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
    'BertzCT'          : Descriptors.BertzCT,
    'QED'              : QED.qed,
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

def calc_inhibitor_similarity(mol, inhibitor_mols):
    """알려진 저해제들과의 Tanimoto 유사도 계산"""
    if mol is None:
        return [0.0] * len(inhibitor_mols)
    
    try:
        fp_query = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        similarities = []
        
        for inhibitor_mol in inhibitor_mols.values():
            if inhibitor_mol is not None:
                fp_ref = AllChem.GetMorganFingerprintAsBitVect(inhibitor_mol, 2)
                similarity = DataStructs.TanimotoSimilarity(fp_query, fp_ref)
                similarities.append(similarity)
            else:
                similarities.append(0.0)
        
        return similarities
    except:
        return [0.0] * len(inhibitor_mols)

# Known inhibitor 분자들 미리 변환
inhibitor_mols = {}
for name, smiles in KNOWN_INHIBITORS.items():
    inhibitor_mols[name] = mol_from_smiles(smiles)

# ------------------------- 5. Feature engineering ---------------------------
def featurize_dataframe(df: pd.DataFrame, smiles_col='Canonical_Smiles'):
    """분자 특성 + 저해제 유사도 계산"""
    mols = df[smiles_col].apply(mol_from_smiles)
    
    # 기본 특성 계산
    desc_mat = np.vstack(mols.apply(calc_descriptors).values)
    desc_cols = list(CORE_DESCRIPTOR_FUNCS.keys())
    desc_df = pd.DataFrame(desc_mat, columns=desc_cols, index=df.index)
    
    # 저해제 유사도 계산
    sim_mat = np.vstack(mols.apply(lambda m: calc_inhibitor_similarity(m, inhibitor_mols)).values)
    sim_cols = [f'similarity_{name}' for name in KNOWN_INHIBITORS.keys()]
    sim_df = pd.DataFrame(sim_mat, columns=sim_cols, index=df.index)
    
    # 최대 유사도 추가
    sim_df['max_inhibitor_similarity'] = sim_df.max(axis=1)
    
    return pd.concat([desc_df, sim_df], axis=1)

print('분자 특성 + 저해제 유사도 생성 중...')
X_train_raw = featurize_dataframe(train_df)
X_test_raw  = featurize_dataframe(test_df)
y_train = train_df['Inhibition']

print(f'생성된 특성 수: {X_train_raw.shape[1]} (기본 {len(CORE_DESCRIPTOR_FUNCS)}개 + 유사도 {len(KNOWN_INHIBITORS)+1}개)')

# ------------------------- 6. NaN 처리 ------------------------------------
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

# ------------------------- 7. 스케일링 ------------------------------------
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

# ------------------------- 8. 최종 데이터 준비 -----------------------------
X_train = X_train_scaled.copy()
X_test = X_test_scaled.copy()

# AutoGluon용 타겟 변수 추가
X_train['Inhibition'] = y_train.values

print(f'최종 특성 행렬: {X_train.shape[1]-1}개 특성; '
      f'{X_train.shape[0]}개 훈련 샘플 / {X_test.shape[0]}개 테스트 샘플')

# ------------------------- 9. 저장 ----------------------------------------
os.makedirs('./output/try11', exist_ok=True)
X_train.to_csv('./output/try11/preprocessed_train.csv', index=False)
X_test.to_csv('./output/try11/preprocessed_test.csv', index=False)

# 특성 정보 저장
feature_info = {
    'feature_count': len(X_train.columns) - 1,
    'feature_names': [col for col in X_train.columns if col != 'Inhibition'],
    'scaling_method': 'StandardScaler',
    'known_inhibitors_used': list(KNOWN_INHIBITORS.keys()),
    'similarity_features': len(KNOWN_INHIBITORS) + 1,
    'description': '핵심 분자특성 + 알려진 CYP3A4 저해제와의 구조적 유사도'
}

with open('./output/try11/feature_selection_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2, ensure_ascii=False)

print("전처리 완료. './output/try11/' 폴더에 저장됨")
print(f"추가된 유사도 특성: {list(KNOWN_INHIBITORS.keys())} + max_similarity")