# ------------------------- 1. Imports ---------------------------------------
import os, json, warnings
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from sklearn.preprocessing import StandardScaler

# ------------------------- 2. I/O -------------------------------------------
train_df = pd.read_csv('./data/train.csv')
test_df  = pd.read_csv('./data/test.csv')

# ------------------------- 3. Helper functions ------------------------------
# 3-A  Classical (scalar) molecular descriptors you usually see in papers
BASIC_DESCRIPTOR_FUNCS = {
    'MolWt'            : Descriptors.MolWt,
    'MolLogP'          : Descriptors.MolLogP,
    'TPSA'             : rdMolDescriptors.CalcTPSA,
    'NumHBD'           : rdMolDescriptors.CalcNumLipinskiHBD,
    'NumHBA'           : rdMolDescriptors.CalcNumLipinskiHBA,
    'NumRotatableBonds': Descriptors.NumRotatableBonds,
    'NumRings'         : Descriptors.RingCount,
    'FractionCSP3'     : rdMolDescriptors.CalcFractionCSP3,
    'HeavyAtomCount'   : rdMolDescriptors.CalcNumHeavyAtoms,
}

def mol_from_smiles(smi: str):
    """Convert SMILES → RDKit Mol, returns None if invalid."""
    mol = Chem.MolFromSmiles(smi)
    return mol

def calc_basic_desc(mol):
    """Return list of scalar descriptors for a Mol (NaN if mol is None)."""
    if mol is None:
        return [np.nan]*len(BASIC_DESCRIPTOR_FUNCS)
    return [f(mol) for f in BASIC_DESCRIPTOR_FUNCS.values()]

# 3-B  Bit fingerprints (circular ECFP/Morgan)
FP_SIZE  = 1024   # You may choose 2048; 1024 keeps file size small
FP_RADIUS = 2

def calc_morgan_fp(mol):
    """Return np.array of ints (0/1) of length FP_SIZE."""
    if mol is None:
        return np.zeros(FP_SIZE, dtype=np.uint8)
    bitvect = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=FP_RADIUS, nBits=FP_SIZE, useChirality=True)
    arr = np.frombuffer(bitvect.ToBitString().encode('ascii'), 'u1') - ord('0')
    return arr  # shape (FP_SIZE,)

# ------------------------- 4. Feature engineering ---------------------------
def featurize_dataframe(df: pd.DataFrame, smiles_col='Canonical_Smiles'):
    """
    Converts a dataframe with SMILES into descriptor + fingerprint features.
    Returns a new dataframe with the same index.
    """
    mols = df[smiles_col].apply(mol_from_smiles)

    # 4-A compute classical descriptors
    desc_mat = np.vstack(mols.apply(calc_basic_desc).values)
    desc_cols = list(BASIC_DESCRIPTOR_FUNCS.keys())
    desc_df = pd.DataFrame(desc_mat, columns=desc_cols, index=df.index)

    # 4-B compute fingerprints (0/1 integers)
    fp_mat = np.vstack(mols.apply(calc_morgan_fp).values)
    fp_cols = [f'fp_{i}' for i in range(FP_SIZE)]
    fp_df = pd.DataFrame(fp_mat, columns=fp_cols, index=df.index).astype(np.uint8)

    return pd.concat([desc_df, fp_df], axis=1)

print('Generating numeric features … (this ~1-2 s per 10k molecules)')
X_train_raw = featurize_dataframe(train_df)
X_test_raw  = featurize_dataframe(test_df)
y_train = train_df['Inhibition']

# ------------------------- 5. Dealing with NaNs -----------------------------
# (invalid SMILES produce NaNs in scalar descriptors; fingerprints are all 0)
nan_rows = X_train_raw.isna().any(axis=1).sum()
if nan_rows:
    warnings.warn(f'{nan_rows} training molecules had invalid SMILES; they will be dropped')
    keep_idx = ~X_train_raw.isna().any(axis=1)
    X_train_raw = X_train_raw.loc[keep_idx]
    y_train     = y_train.loc[keep_idx]

# Do the same for test set (we can simply impute with 0 because the number is tiny)
X_test_raw.fillna(0, inplace=True)

# ------------------------- 6. Scaling ---------------------------------------
# By convention fingerprints (0/1 bits) are left untouched; only scale the scalar 9 columns
scalar_cols = list(BASIC_DESCRIPTOR_FUNCS.keys())
fp_cols     = [c for c in X_train_raw.columns if c.startswith('fp_')]

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
X_train = pd.concat([X_train_scaled_scalar, X_train_raw[fp_cols]], axis=1)
X_train['Inhibition'] = y_train
X_test  = pd.concat([X_test_scaled_scalar,  X_test_raw[fp_cols]], axis=1)

print(f'Final feature matrix: {X_train.shape[1]} columns; '
      f'{X_train.shape[0]} training rows / {X_test.shape[0]} test rows.')

# ------------------------- 7. (optional) save to disk -----------------------
X_train.to_csv('./output/try5/preprocessed_train.csv', index=False)
X_test .to_csv('./output/try5/preprocessed_test.csv' , index=False)