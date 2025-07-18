# ------------------------- 1. Enhanced Imports ------------------------------
import os, json, warnings
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.decomposition import PCA

# ------------------------- 2. I/O -------------------------------------------
train_df = pd.read_csv('./data/train.csv')
test_df  = pd.read_csv('./data/test.csv')

# ------------------------- 3. Enhanced Helper Functions ---------------------
# 3-A Extended molecular descriptors
EXTENDED_DESCRIPTOR_FUNCS = {
    'MolWt': Descriptors.MolWt,
    'MolLogP': Descriptors.MolLogP,
    'TPSA': rdMolDescriptors.CalcTPSA,
    'NumHBD': rdMolDescriptors.CalcNumLipinskiHBD,
    'NumHBA': rdMolDescriptors.CalcNumLipinskiHBA,
    'NumRotatableBonds': Descriptors.NumRotatableBonds,
    'NumRings': Descriptors.RingCount,
    'FractionCSP3': rdMolDescriptors.CalcFractionCSP3,
    'HeavyAtomCount': rdMolDescriptors.CalcNumHeavyAtoms,
    'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings,
    'NumAliphaticRings': rdMolDescriptors.CalcNumAliphaticRings,
    'NumHeterocycles': Descriptors.NumHeteroatoms,
    'Charge': Descriptors.NumRadicalElectrons
}

def mol_from_smiles(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        Chem.SanitizeMol(mol)  # Additional sanitization
    return mol

def calc_extended_desc(mol):
    if mol is None:
        return [np.nan]*len(EXTENDED_DESCRIPTOR_FUNCS)
    return [f(mol) for f in EXTENDED_DESCRIPTOR_FUNCS.values()]

# 3-B Hybrid fingerprint approach (bit-based + count-based)
FP_SIZE = 1024
FP_RADIUS = 2

def calc_hybrid_fingerprints(mol):
    if mol is None:
        return {
            'morgan_bit': np.zeros(FP_SIZE, dtype=np.uint8),
            'morgan_count': np.zeros(FP_SIZE, dtype=np.uint16)
        }
    
    # Bit-based fingerprint
    bit_fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=FP_RADIUS, nBits=FP_SIZE, useChirality=True)
    bit_arr = np.frombuffer(bit_fp.ToBitString().encode('ascii'), 'u1') - ord('0')
    
    # Count-based fingerprint
    count_fp = AllChem.GetMorganFingerprint(mol, radius=FP_RADIUS, useChirality=True)
    count_arr = np.zeros(FP_SIZE, dtype=np.uint16)
    for bit_id, count in count_fp.GetNonzeroElements().items():
        if bit_id < FP_SIZE:
            count_arr[bit_id] = min(count, 255)
    
    return {'morgan_bit': bit_arr, 'morgan_count': count_arr}

# ------------------------- 4. Advanced Feature Engineering ------------------
def featurize_dataframe(df: pd.DataFrame, smiles_col='Canonical_Smiles'):
    mols = df[smiles_col].apply(mol_from_smiles)
    
    # Extended descriptors
    desc_mat = np.vstack(mols.apply(calc_extended_desc).values)
    desc_cols = list(EXTENDED_DESCRIPTOR_FUNCS.keys())
    desc_df = pd.DataFrame(desc_mat, columns=desc_cols, index=df.index)
    
    # Hybrid fingerprints
    fps = mols.apply(calc_hybrid_fingerprints)
    bit_mat = np.vstack([fp['morgan_bit'] for fp in fps])
    count_mat = np.vstack([fp['morgan_count'] for fp in fps])
    
    bit_cols = [f'morgan_bit_{i}' for i in range(FP_SIZE)]
    count_cols = [f'morgan_count_{i}' for i in range(FP_SIZE)]
    
    bit_df = pd.DataFrame(bit_mat, columns=bit_cols, index=df.index).astype(np.uint8)
    count_df = pd.DataFrame(count_mat, columns=count_cols, index=df.index).astype(np.uint16)
    
    return pd.concat([desc_df, bit_df, count_df], axis=1)

print('Generating enhanced hybrid features...')
X_train_raw = featurize_dataframe(train_df)
X_test_raw  = featurize_dataframe(test_df)
y_train = train_df['Inhibition']

# ------------------------- 5. Handling Invalid SMILES -----------------------
nan_rows = X_train_raw.isna().any(axis=1).sum()
if nan_rows:
    warnings.warn(f'Dropping {nan_rows} training rows with invalid SMILES')
    valid_idx = ~X_train_raw.isna().any(axis=1)
    X_train_raw = X_train_raw.loc[valid_idx]
    y_train = y_train.loc[valid_idx]

X_test_raw.fillna(0, inplace=True)

# ------------------------- 6. Target-Aware Feature Engineering ---------------
# Use training target mean for test set features
train_target_mean = y_train.mean()
epsilon = 1e-8  # Small value to prevent division by zero

for col in EXTENDED_DESCRIPTOR_FUNCS.keys():
    # For training set: use actual target values with epsilon
    ratio_train = X_train_raw[col] / (y_train + epsilon)
    
    # Clip extreme values to avoid numerical instability
    ratio_train = np.clip(ratio_train, -1e6, 1e6)
    
    # Use logarithmic transformation for better distribution
    X_train_raw[f'{col}_log_target_ratio'] = np.log1p(np.abs(ratio_train)) * np.sign(ratio_train)
    
    # For test set: use training target mean with epsilon
    ratio_test = X_test_raw[col] / (train_target_mean + epsilon)
    ratio_test = np.clip(ratio_test, -1e6, 1e6)
    X_test_raw[f'{col}_log_target_ratio'] = np.log1p(np.abs(ratio_test)) * np.sign(ratio_test)

# ------------------------- 7. Advanced Scaling ------------------------------
scaler = StandardScaler()
scalar_cols = list(EXTENDED_DESCRIPTOR_FUNCS.keys()) + \
              [c for c in X_train_raw.columns if 'log_target_ratio' in c]

X_train_scaled = X_train_raw.copy()
X_test_scaled = X_test_raw.copy()

X_train_scaled[scalar_cols] = scaler.fit_transform(X_train_raw[scalar_cols])
X_test_scaled[scalar_cols] = scaler.transform(X_test_raw[scalar_cols])

# ------------------------- 8. Smart Feature Selection -----------------------
print('Applying advanced feature selection...')

# 8-A Remove near-zero variance features
variance_selector = VarianceThreshold(threshold=0.01)
X_train_var = variance_selector.fit_transform(X_train_scaled)
X_test_var = variance_selector.transform(X_test_scaled)
selected_features = X_train_scaled.columns[variance_selector.get_support()]

# 8-B Mutual information selection
mi_scores = mutual_info_regression(X_train_var, y_train, random_state=42)
mi_threshold = np.percentile(mi_scores, 90)  # Top 10% features
mi_mask = mi_scores >= mi_threshold
X_train_mi = X_train_var[:, mi_mask]
X_test_mi = X_test_var[:, mi_mask]
mi_features = selected_features[mi_mask]

# 8-C PCA for fingerprint compression
pca = PCA(n_components=0.95, random_state=42)
fp_cols = [c for c in mi_features if 'morgan_' in c]
non_fp_cols = [c for c in mi_features if 'morgan_' not in c]

# Apply PCA only to fingerprint features
X_train_fp_pca = pca.fit_transform(X_train_mi[:, [i for i, col in enumerate(mi_features) if col in fp_cols]])
X_test_fp_pca = pca.transform(X_test_mi[:, [i for i, col in enumerate(mi_features) if col in fp_cols]])

# Combine PCA results with non-fingerprint features
X_train_non_fp = X_train_mi[:, [i for i, col in enumerate(mi_features) if col in non_fp_cols]]
X_test_non_fp = X_test_mi[:, [i for i, col in enumerate(mi_features) if col in non_fp_cols]]

X_train_final = np.hstack([X_train_non_fp, X_train_fp_pca])
X_test_final = np.hstack([X_test_non_fp, X_test_fp_pca])

# Create final feature names
pca_cols = [f'fp_pca_{i}' for i in range(X_train_fp_pca.shape[1])]
final_cols = non_fp_cols + pca_cols

# ------------------------- 9. Save Results ----------------------------------
os.makedirs('./output/try4', exist_ok=True)

# Final datasets
X_train = pd.DataFrame(X_train_final, columns=final_cols, index=X_train_raw.index)
X_test = pd.DataFrame(X_test_final, columns=final_cols, index=X_test_raw.index)
X_train['Inhibition'] = y_train.values

X_train.to_csv('./output/try4/preprocessed_train.csv', index=False)
X_test.to_csv('./output/try4/preprocessed_test.csv', index=False)

# Save feature selection info
feature_info = {
    'original_features': int(X_train_raw.shape[1]),  # Convert to Python int
    'after_variance_selection': int(len(selected_features)),  # Convert to Python int
    'after_mi_selection': int(len(mi_features)),  # Convert to Python int
    'pca_components': int(pca.n_components_),  # Convert to Python int
    'final_features': int(len(final_cols))  # Convert to Python int
}

with open('./output/try4/feature_selection_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)

print(f'Final feature matrix: {len(final_cols)} features | '
      f'Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows')
print('Preprocessing complete! Files saved to ./output/try4/')
