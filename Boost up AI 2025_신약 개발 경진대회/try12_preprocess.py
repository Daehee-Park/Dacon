# ------------------------- 1. Imports ---------------------------------------
import os, json, warnings
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, QED, AllChem, Crippen, Lipinski

from sklearn.preprocessing import StandardScaler

# ------------------------- 2. I/O -------------------------------------------
train_df = pd.read_csv('./data/train.csv')
test_df  = pd.read_csv('./data/test.csv')

# ------------------------- 3. Domain Knowledge Definitions ------------------
# 3-A: 알려진 저해제 (PubChem Canonical SMILES로 검증 및 수정)
KNOWN_INHIBITORS = {
    # PubChem CID: 5790
    'ketoconazole': 'CC(=O)N1CCN(CC1)C2=CC=C(C=C2)OC3CCN(CC3)C(=O)C(C)(C)OC4=CC=C(C=C4)Cl',
    # PubChem CID: 2354
    'bergapten': 'COC1=CC2=C(C=C1)C(=O)OC3=C2C=CC(=C3)O',
    # PubChem CID: 5280343
    'quercetin': 'C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O',
    # PubChem CID: 638024
    'piperine': 'C1CCN(CC1)C(=O)C=CC=C2C=CC3=C(C2)OCO3'
}

# 3-B: 구조적 경고 (CYP3A4 저해 관련 Pharmacophores - SMARTS)
STRUCTURAL_ALERTS = {
    'furanocoumarin': Chem.MolFromSmarts('o1c2ccccc2oc1=O'), # Bergapten-like
    'imidazole': Chem.MolFromSmarts('c1cncn1'),              # Ketoconazole-like
    'catechol': Chem.MolFromSmarts('c1c(O)c(O)ccc1'),         # Quercetin-like
    'basic_nitrogen': Chem.MolFromSmarts('[#7;+0;D2,D3]'),    # Piperine-like basic nitrogen
}

# ------------------------- 4. Helper functions ------------------------------
# 4-A: 특성 계산 함수들
def mol_from_smiles(smi: str):
    return Chem.MolFromSmiles(smi)

def calc_all_descriptors(mol):
    """모든 물리화학적/구조적 특성 계산"""
    if mol is None: return [np.nan] * 16 # 특성 개수
    
    return [
        Descriptors.MolWt(mol), rdMolDescriptors.CalcTPSA(mol),
        rdMolDescriptors.CalcNumLipinskiHBD(mol), rdMolDescriptors.CalcNumLipinskiHBA(mol),
        Descriptors.NumRotatableBonds(mol), Descriptors.RingCount(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol), rdMolDescriptors.CalcNumHeavyAtoms(mol),
        rdMolDescriptors.CalcFractionCSP3(mol), Descriptors.BertzCT(mol), QED.qed(mol),
        Crippen.MolLogP(mol), Crippen.MolMR(mol), # Crippen Descriptors
        rdMolDescriptors.CalcLabuteASA(mol), # Labute ASA
        Lipinski.NumAliphaticRings(mol), Lipinski.NumAromaticHeterocycles(mol)
    ]

def calc_inhibitor_similarity(mol, inhibitor_fps):
    """알려진 저해제들과의 Tanimoto 유사도 계산"""
    if mol is None: return [0.0] * len(inhibitor_fps)
    try:
        fp_query = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return [DataStructs.TanimotoSimilarity(fp_query, fp_ref) for fp_ref in inhibitor_fps]
    except: return [0.0] * len(inhibitor_fps)
        
def count_structural_alerts(mol, alert_mols):
    """구조적 경고(Pharmacophore) 포함 여부 확인"""
    if mol is None: return [0] * len(alert_mols)
    return [1 if mol.HasSubstructMatch(alert) else 0 for alert in alert_mols]

# 4-B: 계산 가속화를 위한 사전 처리
inhibitor_mols = {name: mol_from_smiles(smi) for name, smi in KNOWN_INHIBITORS.items()}
inhibitor_fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048) for m in inhibitor_mols.values() if m is not None]
alert_mols = list(STRUCTURAL_ALERTS.values())

# ------------------------- 5. Feature engineering ---------------------------
def featurize_dataframe(df: pd.DataFrame, smiles_col='Canonical_Smiles'):
    """도메인 지식 기반 특성 공학 파이프라인"""
    mols = df[smiles_col].apply(mol_from_smiles)

    # 5-A: 물리화학적 특성
    desc_names = ['MolWt','TPSA','NumHBD','NumHBA','NumRotBonds','NumRings','NumAroRings',
                  'HeavyAtomCount','FracCSP3','BertzCT','QED','MolLogP','MolMR',
                  'LabuteASA','NumAliphaticRings','NumAroHeterocycles']
    desc_df = pd.DataFrame(np.vstack(mols.apply(calc_all_descriptors).values), 
                           columns=desc_names, index=df.index)

    # 5-B: 알려진 저해제와의 유사도
    sim_names = [f'sim_{name}' for name in KNOWN_INHIBITORS.keys()]
    sim_df = pd.DataFrame(np.vstack(mols.apply(lambda m: calc_inhibitor_similarity(m, inhibitor_fps)).values), 
                          columns=sim_names, index=df.index)
    sim_df['max_inhibitor_similarity'] = sim_df.max(axis=1)

    # 5-C: 구조적 경고 (Pharmacophores)
    alert_names = [f'alert_{name}' for name in STRUCTURAL_ALERTS.keys()]
    alert_df = pd.DataFrame(np.vstack(mols.apply(lambda m: count_structural_alerts(m, alert_mols)).values),
                            columns=alert_names, index=df.index)

    # 5-D: 모든 특성 결합
    full_df = pd.concat([desc_df, sim_df, alert_df], axis=1)

    # 5-E: 비율 특성 (Ratio Features)
    full_df['TPSA_ratio'] = (full_df['TPSA'] / full_df['MolWt']).fillna(0)
    full_df['LogP_per_Heavy'] = (full_df['MolLogP'] / full_df['HeavyAtomCount']).fillna(0)

    return full_df

print('도메인 지식 기반 고급 특성 생성 중...')
X_train_raw = featurize_dataframe(train_df)
X_test_raw  = featurize_dataframe(test_df)
y_train = train_df['Inhibition']

print(f'생성된 총 특성 수: {X_train_raw.shape[1]}')

# ------------------------- 6. NaN 처리 및 스케일링 --------------------------
X_train_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test_raw.replace([np.inf, -np.inf], np.nan, inplace=True)

nan_rows = X_train_raw.isna().any(axis=1).sum()
if nan_rows > 0:
    warnings.warn(f'{nan_rows}개 분자에서 특성 계산 실패; 제거됩니다')
    keep_idx = ~X_train_raw.isna().any(axis=1)
    X_train_raw = X_train_raw.loc[keep_idx]
    y_train     = y_train.loc[keep_idx]

if X_test_raw.isna().any().any():
    imputation_values = X_train_raw.median()
    X_test_raw.fillna(imputation_values, inplace=True)
    X_test_raw.fillna(0, inplace=True)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns, index=X_train_raw.index)
X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns, index=X_test_raw.index)

# ------------------------- 7. 최종 데이터 준비 및 저장 ----------------------
X_train['Inhibition'] = y_train.values

print(f'최종 특성 행렬: {X_train.shape[1]-1}개 특성; '
      f'{X_train.shape[0]}개 훈련 샘플 / {X_test.shape[0]}개 테스트 샘플')

os.makedirs('./output/try12', exist_ok=True)
X_train.to_csv('./output/try12/preprocessed_train.csv', index=False)
X_test.to_csv('./output/try12/preprocessed_test.csv', index=False)

feature_info = {
    'total_features': len(X_train.columns) - 1,
    'feature_categories': {
        'descriptors': 16,
        'similarity_features': len(KNOWN_INHIBITORS) + 1,
        'structural_alerts': len(STRUCTURAL_ALERTS),
        'ratio_features': 2
    },
    'feature_names': [col for col in X_train.columns if col != 'Inhibition'],
    'description': '의화학 도메인 지식 기반 특성: 고급 물리화학적 특성, 알려진 저해제 유사도, 구조적 경고(Pharmacophore), 비율 특성 포함'
}
with open('./output/try12/feature_selection_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2, ensure_ascii=False)

print("전처리 완료. './output/try12/' 폴더에 저장됨") 