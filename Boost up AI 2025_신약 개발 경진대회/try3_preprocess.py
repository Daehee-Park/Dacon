# ------------------------- 1. Imports ---------------------------------------
import os, json, warnings
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor

# ------------------------- 2. I/O -------------------------------------------
train_df = pd.read_csv('./data/train.csv')
test_df  = pd.read_csv('./data/test.csv')

# ------------------------- 3. Helper functions ------------------------------
# 3-A  Classical (scalar) molecular descriptors
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

# 3-B  Count-based Morgan fingerprints
FP_SIZE  = 2048   # Increased size for count-based fingerprints
FP_RADIUS = 2

def calc_morgan_count_fp(mol):
    """Return count-based Morgan fingerprint as np.array."""
    if mol is None:
        return np.zeros(FP_SIZE, dtype=np.uint16)
    
    # Get count-based fingerprint as dictionary {bit_id: count}
    fp_dict = AllChem.GetMorganFingerprint(mol, radius=FP_RADIUS, useChirality=True)
    
    # Convert to fixed-size array
    fp_array = np.zeros(FP_SIZE, dtype=np.uint16)
    for bit_id, count in fp_dict.GetNonzeroElements().items():
        if bit_id < FP_SIZE:
            fp_array[bit_id] = min(count, 255)  # Cap at 255 to prevent overflow
    
    return fp_array

# 3-C  MACCS Keys fingerprints (167 bits)
def calc_maccs_fp(mol):
    """Return MACCS Keys fingerprint as np.array."""
    if mol is None:
        return np.zeros(167, dtype=np.uint8)
    
    maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
    # Convert to numpy array
    arr = np.frombuffer(maccs_fp.ToBitString().encode('ascii'), 'u1') - ord('0')
    return arr.astype(np.uint8)

# 3-D  Atom Pair fingerprints (additional fingerprint type)
def calc_atompair_fp(mol, fp_size=1024):
    """Return Atom Pair fingerprint as np.array."""
    if mol is None:
        return np.zeros(fp_size, dtype=np.uint16)
    
    ap_fp = AllChem.GetAtomPairFingerprint(mol)
    fp_array = np.zeros(fp_size, dtype=np.uint16)
    
    for bit_id, count in ap_fp.GetNonzeroElements().items():
        if bit_id < fp_size:
            fp_array[bit_id] = min(count, 255)
    
    return fp_array

# ------------------------- 4. Feature engineering ---------------------------
def featurize_dataframe(df: pd.DataFrame, smiles_col='Canonical_Smiles'):
    """
    Converts a dataframe with SMILES into descriptor + multiple fingerprint features.
    Returns a new dataframe with the same index.
    """
    mols = df[smiles_col].apply(mol_from_smiles)

    # 4-A compute classical descriptors
    desc_mat = np.vstack(mols.apply(calc_basic_desc).values)
    desc_cols = list(BASIC_DESCRIPTOR_FUNCS.keys())
    desc_df = pd.DataFrame(desc_mat, columns=desc_cols, index=df.index)

    # 4-B compute count-based Morgan fingerprints
    morgan_mat = np.vstack(mols.apply(calc_morgan_count_fp).values)
    morgan_cols = [f'morgan_{i}' for i in range(FP_SIZE)]
    morgan_df = pd.DataFrame(morgan_mat, columns=morgan_cols, index=df.index).astype(np.uint16)

    # 4-C compute MACCS Keys fingerprints
    maccs_mat = np.vstack(mols.apply(calc_maccs_fp).values)
    maccs_cols = [f'maccs_{i}' for i in range(167)]  # Changed from 166 to 167
    maccs_df = pd.DataFrame(maccs_mat, columns=maccs_cols, index=df.index).astype(np.uint8)

    # 4-D compute Atom Pair fingerprints
    ap_mat = np.vstack(mols.apply(lambda m: calc_atompair_fp(m, 512)).values)
    ap_cols = [f'atompair_{i}' for i in range(512)]
    ap_df = pd.DataFrame(ap_mat, columns=ap_cols, index=df.index).astype(np.uint16)

    return pd.concat([desc_df, morgan_df, maccs_df, ap_df], axis=1)

print('Generating enhanced features (descriptors + count-based fingerprints + MACCS + AtomPair)...')
X_train_raw = featurize_dataframe(train_df)
X_test_raw  = featurize_dataframe(test_df)
y_train = train_df['Inhibition']

print(f'Raw feature matrix: {X_train_raw.shape[1]} features')

# ------------------------- 5. Dealing with NaNs -----------------------------
nan_rows = X_train_raw.isna().any(axis=1).sum()
if nan_rows:
    warnings.warn(f'{nan_rows} training molecules had invalid SMILES; they will be dropped')
    keep_idx = ~X_train_raw.isna().any(axis=1)
    X_train_raw = X_train_raw.loc[keep_idx]
    y_train     = y_train.loc[keep_idx]

X_test_raw.fillna(0, inplace=True)

# ------------------------- 6. Scaling ---------------------------------------
# Scale only the classical descriptors; fingerprints are left as-is
scalar_cols = list(BASIC_DESCRIPTOR_FUNCS.keys())
fp_cols = [c for c in X_train_raw.columns if not c in scalar_cols]

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
X_train_full = pd.concat([X_train_scaled_scalar, X_train_raw[fp_cols]], axis=1)
X_test_full  = pd.concat([X_test_scaled_scalar,  X_test_raw[fp_cols]], axis=1)

# ------------------------- 7. Feature Selection -----------------------------
print('Applying feature selection...')

# 7-A Remove low-variance features
variance_selector = VarianceThreshold(threshold=0.01)  # Remove features with very low variance
X_train_var = variance_selector.fit_transform(X_train_full)
X_test_var = variance_selector.transform(X_test_full)

selected_features = X_train_full.columns[variance_selector.get_support()]
print(f'After variance threshold: {len(selected_features)} features')

# 7-B Select top K features using univariate feature selection
k_best = min(1000, len(selected_features))  # Select top 1000 or all if less
univariate_selector = SelectKBest(score_func=f_regression, k=k_best)
X_train_univariate = univariate_selector.fit_transform(X_train_var, y_train)
X_test_univariate = univariate_selector.transform(X_test_var)

univariate_features = selected_features[univariate_selector.get_support()]
print(f'After univariate selection: {len(univariate_features)} features')

# 7-C Random Forest feature importance selection
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_univariate, y_train)

# Get feature importance and select top features
importance_scores = rf.feature_importances_
importance_threshold = np.percentile(importance_scores, 75)  # Top 25% by importance
important_mask = importance_scores >= importance_threshold

X_train_final = X_train_univariate[:, important_mask]
X_test_final = X_test_univariate[:, important_mask]
final_features = univariate_features[important_mask]

print(f'After RF importance selection: {X_train_final.shape[1]} features')

# Convert back to DataFrame with proper column names
X_train = pd.DataFrame(X_train_final, columns=final_features, index=X_train_full.index)
X_test = pd.DataFrame(X_test_final, columns=final_features, index=X_test_full.index)

# Add target for train set
X_train['Inhibition'] = y_train

print(f'Final feature matrix: {X_train.shape[1]-1} features; '
      f'{X_train.shape[0]} training rows / {X_test.shape[0]} test rows.')

# ------------------------- 8. Save results ----------------------------------
os.makedirs('./output/try3', exist_ok=True)

# Save processed data
X_train.to_csv('./output/try3/preprocessed_train.csv', index=False)
X_test.to_csv('./output/try3/preprocessed_test.csv', index=False)

# Save feature selection info
feature_info = {
    'original_features': len(X_train_full.columns),
    'after_variance_threshold': len(selected_features),
    'after_univariate_selection': len(univariate_features),
    'final_features': len(final_features),
    'final_feature_names': final_features.tolist(),
    'feature_selection_params': {
        'variance_threshold': 0.01,
        'univariate_k': k_best,
        'rf_importance_percentile': 75
    }
}

with open('./output/try3/feature_selection_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

# Save feature importance scores
importance_df = pd.DataFrame({
    'feature': final_features,
    'importance': importance_scores[important_mask]
}).sort_values('importance', ascending=False)

importance_df.to_csv('./output/try3/feature_importance.csv', index=False)

print('Preprocessing completed! Files saved to ./output/try3/') 