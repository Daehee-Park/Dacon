# ------------------------- 1. Imports ---------------------------------------
import os, json, warnings
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# ------------------------- 2. I/O -------------------------------------------
train_df = pd.read_csv('./data/train.csv')
test_df  = pd.read_csv('./data/test.csv')

# ------------------------- 3. Helper functions ------------------------------
# 3-A  Classical (scalar) molecular descriptors
COMPREHENSIVE_DESCRIPTOR_FUNCS = {
    'MolWt'            : Descriptors.MolWt,
    'MolLogP'          : Descriptors.MolLogP,
    'TPSA'             : rdMolDescriptors.CalcTPSA,
    'NumHBD'           : rdMolDescriptors.CalcNumLipinskiHBD,
    'NumHBA'           : rdMolDescriptors.CalcNumLipinskiHBA,
    'NumRotatableBonds': Descriptors.NumRotatableBonds,
    'NumRings'         : Descriptors.RingCount,
    'FractionCSP3'     : rdMolDescriptors.CalcFractionCSP3,
    'HeavyAtomCount'   : rdMolDescriptors.CalcNumHeavyAtoms,
    'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings,
    'NumAromaticRings' : rdMolDescriptors.CalcNumAromaticRings,
    'MolMR'            : Descriptors.MolMR,
    'BalabanJ'         : Descriptors.BalabanJ,
}

def mol_from_smiles(smi: str):
    """Convert SMILES → RDKit Mol, returns None if invalid."""
    mol = Chem.MolFromSmiles(smi)
    return mol

def calc_basic_desc(mol):
    """Return list of scalar descriptors for a Mol (NaN if mol is None)."""
    if mol is None:
        return [np.nan]*len(COMPREHENSIVE_DESCRIPTOR_FUNCS)
    return [f(mol) for f in COMPREHENSIVE_DESCRIPTOR_FUNCS.values()]

# 3-B  Count-based Morgan fingerprints
FP_SIZE  = 2048
FP_RADIUS = 2

def calc_morgan_count_fp(mol):
    """Return count-based Morgan fingerprint as np.array."""
    if mol is None:
        return np.zeros(FP_SIZE, dtype=np.uint16)
    
    fp_dict = AllChem.GetMorganFingerprint(mol, radius=FP_RADIUS, useChirality=True)
    
    fp_array = np.zeros(FP_SIZE, dtype=np.uint16)
    for bit_id, count in fp_dict.GetNonzeroElements().items():
        if bit_id < FP_SIZE:
            fp_array[bit_id] = min(count, 255)
    
    return fp_array

# ------------------------- 4. Feature engineering ---------------------------
def featurize_dataframe(df: pd.DataFrame, smiles_col='Canonical_Smiles'):
    """
    Converts a dataframe with SMILES into descriptor + fingerprint features.
    """
    mols = df[smiles_col].apply(mol_from_smiles)

    # 4-A compute classical descriptors
    desc_mat = np.vstack(mols.apply(calc_basic_desc).values)
    desc_cols = list(COMPREHENSIVE_DESCRIPTOR_FUNCS.keys())
    desc_df = pd.DataFrame(desc_mat, columns=desc_cols, index=df.index)

    # 4-B compute count-based fingerprints
    fp_mat = np.vstack(mols.apply(calc_morgan_count_fp).values)
    fp_cols = [f'fp_{i}' for i in range(FP_SIZE)]
    fp_df = pd.DataFrame(fp_mat, columns=fp_cols, index=df.index).astype(np.uint16)

    return pd.concat([desc_df, fp_df], axis=1)

print('Generating numeric features... (this can take a few minutes)')
X_train_raw = featurize_dataframe(train_df)
X_test_raw  = featurize_dataframe(test_df)
y_train = train_df['Inhibition']

# ------------------------- 5. Dealing with NaNs -----------------------------
nan_rows = X_train_raw.isna().any(axis=1).sum()
if nan_rows > 0:
    warnings.warn(f'{nan_rows} training molecules had invalid SMILES; they will be dropped')
    keep_idx = ~X_train_raw.isna().any(axis=1)
    X_train_raw = X_train_raw.loc[keep_idx]
    y_train     = y_train.loc[keep_idx]

# For test set, impute with training data's median
if X_test_raw.isna().any().any():
    imputation_values = X_train_raw.median()
    X_test_raw.fillna(imputation_values, inplace=True)
    X_test_raw.fillna(0, inplace=True) # Fallback for any remaining NaNs

# ------------------------- 6. Scaling ---------------------------------------
scalar_cols = list(COMPREHENSIVE_DESCRIPTOR_FUNCS.keys())
fp_cols = [c for c in X_train_raw.columns if c.startswith('fp_')]

scaler = StandardScaler()
X_train_scaled_scalar = pd.DataFrame(
        scaler.fit_transform(X_train_raw[scalar_cols]),
        columns=scalar_cols,
        index=X_train_raw.index)

X_test_scaled_scalar = pd.DataFrame(
        scaler.transform(X_test_raw[scalar_cols]),
        columns=scalar_cols,
        index=X_test_raw.index)

# Reassemble: scaled scalar part + original fingerprints
X_train_scaled = pd.concat([X_train_scaled_scalar, X_train_raw[fp_cols]], axis=1)
X_test_scaled  = pd.concat([X_test_scaled_scalar,  X_test_raw[fp_cols]], axis=1)

# ------------------------- 7. Feature Selection -----------------------------
print('Applying univariate feature selection...')
k_best = 30
univariate_selector = SelectKBest(score_func=f_regression, k=k_best)

X_train_final = univariate_selector.fit_transform(X_train_scaled, y_train)
X_test_final = univariate_selector.transform(X_test_scaled)

final_features = X_train_scaled.columns[univariate_selector.get_support()]
print(f'After univariate selection: {len(final_features)} features')

X_train = pd.DataFrame(X_train_final, columns=final_features, index=X_train_scaled.index)
X_test = pd.DataFrame(X_test_final, columns=final_features, index=X_test_scaled.index)

# Add target variable back for AutoGluon
X_train['Inhibition'] = y_train.values

print(f'Final feature matrix: {X_train.shape[1]-1} features; '
      f'{X_train.shape[0]} training rows / {X_test.shape[0]} test rows.')

# ------------------------- 8. Save to disk ----------------------------------
os.makedirs('./output/try9', exist_ok=True)
X_train.to_csv('./output/try9/preprocessed_train.csv', index=False)
X_test .to_csv('./output/try9/preprocessed_test.csv' , index=False)

# Save feature selection info
feature_info = {
    'original_feature_count': len(X_train_raw.columns),
    'final_feature_count': len(final_features),
    'final_feature_names': final_features.tolist(),
    'k_best': k_best,
}
with open('./output/try9/feature_selection_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print("Preprocessing completed. Files saved to './output/try9/'.") 