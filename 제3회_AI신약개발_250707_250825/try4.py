import os
import json
import random
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from autogluon.tabular import TabularPredictor
from custom_metrics import competition_scorer, pIC50_to_IC50


CFG: Dict[str, Any] = {
    'NBITS': 1024,        # fingerprint size (preprocessing representation)
    'SEED': 33,
    'CPUS': 64,
    'TIME_LIMIT_SEC': 3600 * 8,
    'MEMORY_LIMIT_GB': 256,
    'DESC_VAR_THRESH': 1e-10,   # remove near-constant descriptor features
    'DESC_CORR_THRESH': 0.98,   # drop one of highly correlated descriptor pairs
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(CFG['SEED'])

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'try4')
os.makedirs(OUTPUT_DIR, exist_ok=True)

REC_PATH = os.path.join(OUTPUT_DIR, 'preprocess_recommendations.json')
if os.path.exists(REC_PATH):
    try:
        with open(REC_PATH, 'r', encoding='utf-8') as f:
            rec = json.load(f)
        if 'fingerprint' in rec and isinstance(rec['fingerprint'], dict):
            CFG['NBITS'] = int(rec['fingerprint'].get('nbits', CFG['NBITS']))
        if 'descriptor_variance_threshold' in rec:
            CFG['DESC_VAR_THRESH'] = float(rec['descriptor_variance_threshold'])
        if 'descriptor_corr_threshold' in rec:
            CFG['DESC_CORR_THRESH'] = float(rec['descriptor_corr_threshold'])
    except Exception:
        pass

# ------------------------------- Utils ---------------------------------
def canonicalize_smiles(raw_smiles: str) -> str:
    if not isinstance(raw_smiles, str) or len(raw_smiles) == 0:
        return None
    # remove salts: keep largest fragment by heavy atoms
    fragments = raw_smiles.split('.')
    best_mol = None
    best_heavy = -1
    for frag in fragments:
        mol = Chem.MolFromSmiles(frag)
        if mol is None:
            continue
        heavy = mol.GetNumHeavyAtoms()
        if heavy > best_heavy:
            best_heavy = heavy
            best_mol = mol
    if best_mol is None:
        return None
    can = Chem.MolToSmiles(best_mol, canonical=True)
    return can


def smiles_to_fingerprint(smiles: str, nbits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)
    fp = morgan_gen.GetFingerprint(mol)
    arr = np.zeros((nbits,), dtype=int)
    for i in range(nbits):
        arr[i] = fp.GetBit(i)
    return arr


def calculate_rdkit_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full((len(Descriptors._descList),), np.nan)
    descriptors = [desc_func(mol) for _, desc_func in Descriptors._descList]
    return np.array(descriptors, dtype=float)


def _convert_to_nM(value: float, unit: str) -> float:
    try:
        val = float(value)
    except Exception:
        return np.nan
    if unit is None:
        return val
    u = str(unit).strip().lower()
    if u in ['nm', 'nanomolar', 'nanomolars']:
        return val
    if u in ['um', 'µm', 'micromolar', 'micromolars']:
        return val * 1e3
    if u in ['mm', 'millimolar', 'millimolars']:
        return val * 1e6
    if u in ['pm', 'picomolar', 'picomolars']:
        return val * 1e-3
    # fallback: assume already nM
    return val


def load_and_preprocess_data() -> pd.DataFrame:
    # Load
    chembl = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'ChEMBL_ASK1(IC50).csv'), sep=';')
    pubchem = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'Pubchem_ASK1.csv'), low_memory=False)

    # Clean columns
    chembl.columns = chembl.columns.str.strip().str.replace('"', '')

    # Map to [smiles, ic50_nM]
    chembl = chembl[chembl['Standard Type'] == 'IC50'].copy()
    # Try to handle units if available
    if 'Standard Units' in chembl.columns:
        chembl['ic50_nM'] = chembl.apply(lambda r: _convert_to_nM(r['Standard Value'], r['Standard Units']), axis=1)
    else:
        chembl['ic50_nM'] = pd.to_numeric(chembl['Standard Value'], errors='coerce')
    chembl = chembl.rename(columns={'Smiles': 'smiles'})[['smiles', 'ic50_nM']]

    pubchem = pubchem.rename(columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'})[['smiles', 'ic50_nM']]
    pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')

    df = pd.concat([chembl, pubchem], ignore_index=True)

    # Canonicalize SMILES & remove salts
    df['smiles'] = df['smiles'].astype(str).map(canonicalize_smiles)

    # Basic validity filters
    df = df.dropna(subset=['smiles', 'ic50_nM']).copy()
    df = df[df['ic50_nM'] > 0]

    # Aggregate duplicates by canonical SMILES using median IC50 (robust)
    df = df.groupby('smiles', as_index=False)['ic50_nM'].median()

    return df.reset_index(drop=True)


def build_train_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    # Target
    df = df.copy()
    df['pIC50'] = 9 - np.log10(df['ic50_nM'])

    # Feature generation
    nbits = CFG['NBITS']
    df['fingerprint'] = df['smiles'].apply(lambda s: smiles_to_fingerprint(s, nbits))
    df['descriptors'] = df['smiles'].apply(calculate_rdkit_descriptors)

    df = df.dropna(subset=['fingerprint', 'descriptors']).reset_index(drop=True)

    # Stack features
    fp_stack = np.stack(df['fingerprint'].values)  # [N, nbits]
    desc_stack = np.stack(df['descriptors'].values)  # [N, D]

    # Impute descriptor NaNs by column means (robust)
    col_means = np.nanmean(desc_stack, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    desc_stack = np.where(np.isnan(desc_stack), col_means, desc_stack)

    # Remove near-constant descriptor features
    var_selector = VarianceThreshold(threshold=CFG['DESC_VAR_THRESH'])
    desc_var = var_selector.fit_transform(desc_stack)
    desc_kept_idx_stage1 = var_selector.get_support(indices=True)

    # Remove highly correlated descriptor features
    if desc_var.shape[1] > 1:
        corr = np.corrcoef(desc_var, rowvar=False)
        to_keep_mask = np.ones(corr.shape[0], dtype=bool)
        for i in range(corr.shape[0]):
            if not to_keep_mask[i]:
                continue
            high_corr = np.where(np.abs(corr[i, (i+1):]) > CFG['DESC_CORR_THRESH'])[0]
            high_corr = high_corr + (i + 1)
            to_keep_mask[high_corr] = False
        desc_kept_idx = desc_kept_idx_stage1[to_keep_mask]
    else:
        desc_kept_idx = desc_kept_idx_stage1

    desc_selected = desc_stack[:, desc_kept_idx]

    # Standardize descriptors
    scaler = StandardScaler()
    desc_scaled = scaler.fit_transform(desc_selected)

    # Build final feature matrix and names
    fp_names = [f'fp_{i}' for i in range(fp_stack.shape[1])]
    desc_names_all = [name for name, _ in Descriptors._descList]
    desc_names_stage1 = [desc_names_all[i] for i in desc_kept_idx_stage1]
    desc_names_final = [desc_names_stage1[i] for i, keep in enumerate(to_keep_mask) if desc_var.shape[1] > 1 and keep] if desc_var.shape[1] > 1 else desc_names_stage1

    X = np.hstack([fp_stack, desc_scaled])
    feature_names = fp_names + [f'desc_{n}' for n in desc_names_final]

    train_table = pd.DataFrame(X, columns=feature_names)
    train_table['pIC50'] = df['pIC50'].values

    preprocess_info = {
        'nbits': nbits,
        'desc_col_means': col_means.tolist(),
        'desc_kept_idx_stage1': desc_kept_idx_stage1.tolist(),
        'desc_kept_idx': desc_kept_idx.tolist() if isinstance(desc_kept_idx, np.ndarray) else list(desc_kept_idx),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'feature_names': feature_names,
        'desc_names_final': desc_names_final,
    }

    # Save feature info for reference
    with open(os.path.join(OUTPUT_DIR, 'feature_info.json'), 'w', encoding='utf-8') as f:
        json.dump({k: (v if isinstance(v, (int, float, str, list, dict)) else str(v)) for k, v in preprocess_info.items()}, f, ensure_ascii=False, indent=2)

    # Pack scaler into info object (non-serializable, but used in-process)
    preprocess_info['scaler_obj'] = scaler

    return train_table, df['pIC50'].values, preprocess_info


def build_test_features(test_df: pd.DataFrame, preprocess_info: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series]:
    test_df = test_df.copy()
    nbits = preprocess_info['nbits']

    test_df['Smiles'] = test_df['Smiles'].astype(str).map(canonicalize_smiles)
    test_df = test_df.dropna(subset=['Smiles']).reset_index(drop=True)

    test_df['fingerprint'] = test_df['Smiles'].apply(lambda s: smiles_to_fingerprint(s, nbits))
    test_df['descriptors'] = test_df['Smiles'].apply(calculate_rdkit_descriptors)

    valid_mask = test_df['fingerprint'].notna() & test_df['descriptors'].notna()
    valid_df = test_df[valid_mask].copy()

    if len(valid_df) == 0:
        return pd.DataFrame(columns=preprocess_info['feature_names']), test_df['ID']

    fp_stack = np.stack(valid_df['fingerprint'].values)
    desc_stack = np.stack(valid_df['descriptors'].values)

    # Impute using train means
    col_means = np.asarray(preprocess_info['desc_col_means'])
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    desc_stack = np.where(np.isnan(desc_stack), col_means, desc_stack)

    # Apply same feature selection
    stage1_idx = np.asarray(preprocess_info['desc_kept_idx_stage1'])
    final_idx = np.asarray(preprocess_info['desc_kept_idx'])
    desc_var = desc_stack[:, stage1_idx]
    desc_selected = desc_var[:, np.searchsorted(stage1_idx, final_idx)] if len(final_idx) > 0 else desc_var

    # Scale with train scaler
    scaler = preprocess_info['scaler_obj']
    desc_scaled = scaler.transform(desc_selected)

    X_test = np.hstack([fp_stack, desc_scaled])
    test_table = pd.DataFrame(X_test, columns=preprocess_info['feature_names'])

    valid_df = valid_df.reset_index(drop=True)
    test_table['ID'] = valid_df['ID']

    return test_table, test_df['ID']


def main() -> None:
    print('1) Loading and preprocessing training data...')
    train_df = load_and_preprocess_data()

    print('2) Building train features...')
    train_table, y_train_pic50, preprocess_info = build_train_features(train_df)

    feature_names = preprocess_info['feature_names']
    print(f"Train features: {len(feature_names)} columns; samples: {len(train_table)}")

    print('\n3) Training AutoGluon model...')
    model_path = os.path.join(OUTPUT_DIR, 'autogluon_models')
    if os.path.exists(model_path):
        predictor = TabularPredictor.load(model_path)
        print(f"Loaded existing model from {model_path}")
    else:
        predictor = TabularPredictor(
            label='pIC50',
            eval_metric=competition_scorer,
            path=model_path,
        )
        predictor.fit(
            train_data=train_table,
            time_limit=CFG['TIME_LIMIT_SEC'],
            presets='extreme_quality',
            num_cpus=CFG['CPUS'],
            memory_limit=CFG['MEMORY_LIMIT_GB'],
        )

    print('\n4) Processing test data...')
    test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'test.csv'))
    test_table, all_test_ids = build_test_features(test_df, preprocess_info)

    print('5) Predicting...')
    submission = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'sample_submission.csv'))[['ID']]

    if len(test_table) > 0:
        preds_pic50 = predictor.predict(test_table[feature_names])
        preds_ic50 = pIC50_to_IC50(preds_pic50)
        pred_df = pd.DataFrame({'ID': test_table['ID'].values, 'ASK1_IC50_nM': preds_ic50})
        submission = submission.merge(pred_df, on='ID', how='left')

    # Fallback fill by train mean IC50
    submission['ASK1_IC50_nM'] = submission['ASK1_IC50_nM'].fillna(train_df['ic50_nM'].mean())

    sub_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    submission.to_csv(sub_path, index=False)
    print(f'Submission saved: {sub_path}')

    # Save leaderboard for reference
    leaderboard = predictor.leaderboard(silent=True)
    leaderboard.to_csv(os.path.join(OUTPUT_DIR, 'leaderboard.csv'), index=False)
    print('Leaderboard saved.')


if __name__ == '__main__':
    main()


