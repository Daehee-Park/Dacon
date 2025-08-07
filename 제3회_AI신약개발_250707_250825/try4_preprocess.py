import os
import json
import pickle
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler


# Silence RDKit and misc warnings for clean logs
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


CFG: Dict[str, int] = {
    'NBITS_MORGAN_R2': 2048,
    'NBITS_MORGAN_R3': 1024,
    'NBITS_MACCS': 167,
    'SEED': 33,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(CFG['SEED'])

OUTPUT_DIR = "./output/try4"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_preprocess_data() -> pd.DataFrame:
    """Load ChEMBL and PubChem IC50 datasets and perform basic cleaning."""
    chembl = pd.read_csv("./data/ChEMBL_ASK1(IC50).csv", sep=';')
    pubchem = pd.read_csv("./data/Pubchem_ASK1.csv", low_memory=False)

    chembl.columns = chembl.columns.str.strip().str.replace('"', '')
    chembl = chembl[chembl['Standard Type'] == 'IC50']
    chembl = chembl[['Smiles', 'Standard Value']].rename(
        columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'}
    )
    chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')

    pubchem = pubchem[['SMILES', 'Activity_Value']].rename(
        columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'}
    )
    pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')

    df = pd.concat([chembl, pubchem], ignore_index=True)
    df = df.dropna(subset=['smiles', 'ic50_nM'])
    df = df[df['ic50_nM'] > 0]
    df = df.drop_duplicates(subset='smiles').reset_index(drop=True)
    return df


def robust_ic50_to_pic50(ic50_values: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Convert IC50 (nM) to pIC50 with robust percentile clipping in log space."""
    log_ic50 = np.log10(ic50_values + 1e-9)
    lower = np.percentile(log_ic50, 1)
    upper = np.percentile(log_ic50, 99)
    log_ic50_capped = np.clip(log_ic50, lower, upper)
    ic50_capped = 10 ** log_ic50_capped
    pic50 = 9 - np.log10(ic50_capped + 1e-9)
    return pic50.astype(np.float32), (float(lower), float(upper))


def smiles_to_features(smiles: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract multi-fingerprint features and RDKit descriptors.

    Returns (morgan_r2, morgan_r3, maccs) as bit arrays; descriptors as floats are computed separately.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, None  # type: ignore

    # Morgan radius 2
    morgan2_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=CFG['NBITS_MORGAN_R2'])
    morgan2_fp = morgan2_gen.GetFingerprint(mol)
    morgan2_arr = np.zeros(CFG['NBITS_MORGAN_R2'], dtype=np.int8)
    for i in range(CFG['NBITS_MORGAN_R2']):
        morgan2_arr[i] = morgan2_fp.GetBit(i)

    # Morgan radius 3
    morgan3_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=CFG['NBITS_MORGAN_R3'])
    morgan3_fp = morgan3_gen.GetFingerprint(mol)
    morgan3_arr = np.zeros(CFG['NBITS_MORGAN_R3'], dtype=np.int8)
    for i in range(CFG['NBITS_MORGAN_R3']):
        morgan3_arr[i] = morgan3_fp.GetBit(i)

    # MACCS keys
    maccs_fp = AllChem.GetMACCSKeysFingerprint(mol)
    maccs_arr = np.zeros(CFG['NBITS_MACCS'], dtype=np.int8)
    for i in range(CFG['NBITS_MACCS']):
        maccs_arr[i] = maccs_fp.GetBit(i)

    return morgan2_arr, morgan3_arr, maccs_arr


def _compute_sa_like(mol: Chem.Mol) -> float:
    """간단한 SA-like 지표 (공식 SA 점수 대체).
    고리/브릿지헤드/스파이로 수와 sp3 분율을 조합.
    """
    try:
        n_rings = rdMolDescriptors.CalcNumRings(mol)
        n_bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        frac_sp3 = float(Descriptors.FractionCSP3(mol) or 0.0)
        n_atoms = mol.GetNumAtoms()
        score = (
            0.6 * n_rings + 0.8 * n_bridge + 0.8 * n_spiro + 0.02 * n_atoms - 0.5 * frac_sp3
        )
        return float(score)
    except Exception:
        return float('nan')


def calculate_rdkit_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full((len(Descriptors._descList) + 4,), np.nan, dtype=np.float32)

    base_values = [desc(mol) for _, desc in Descriptors._descList]
    try:
        qed_val = float(Descriptors.qed(mol))
    except Exception:
        qed_val = np.nan
    sa_like = _compute_sa_like(mol)
    n_spiro = float(rdMolDescriptors.CalcNumSpiroAtoms(mol))
    n_bridge = float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    values = base_values + [qed_val, sa_like, n_spiro, n_bridge]
    return np.array(values, dtype=np.float32)


def build_feature_matrix(df: pd.DataFrame, desc_mean: np.ndarray = None, scaler: StandardScaler = None,
                         fit: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, np.ndarray]:
    """Create feature matrix for a dataframe with column 'smiles'.

    Returns (X, valid_mask, desc_mean, scaler, feature_splits)
    where feature_splits stores indices for later reuse: [end_morgan2, end_morgan3, end_maccs]
    """
    # Fingerprints
    fps_r2_list, fps_r3_list, maccs_list, desc_list = [], [], [], []
    valid_mask = []
    for smi in df['smiles']:
        m2, m3, mc = smiles_to_features(smi)
        if m2 is None:
            valid_mask.append(False)
            fps_r2_list.append(None)
            fps_r3_list.append(None)
            maccs_list.append(None)
            desc_list.append(None)
            continue
        valid_mask.append(True)
        fps_r2_list.append(m2)
        fps_r3_list.append(m3)
        maccs_list.append(mc)
        desc_list.append(calculate_rdkit_descriptors(smi))

    valid_mask = np.array(valid_mask)
    if valid_mask.sum() == 0:
        return np.empty((0,)), valid_mask, desc_mean, scaler, np.array([0, 0, 0])  # type: ignore

    fp_r2 = np.stack([x for x, keep in zip(fps_r2_list, valid_mask) if keep])
    fp_r3 = np.stack([x for x, keep in zip(fps_r3_list, valid_mask) if keep])
    maccs = np.stack([x for x, keep in zip(maccs_list, valid_mask) if keep])
    desc = np.stack([x for x, keep in zip(desc_list, valid_mask) if keep])

    # Handle NaNs in descriptors with train mean
    if fit:
        desc_mean = np.nanmean(desc, axis=0)
    desc = np.nan_to_num(desc, nan=desc_mean)

    if fit:
        scaler = StandardScaler()
        desc_scaled = scaler.fit_transform(desc)
    else:
        desc_scaled = scaler.transform(desc)

    X = np.hstack([fp_r2, fp_r3, maccs, desc_scaled]).astype(np.float32)
    end_r2 = fp_r2.shape[1]
    end_r3 = end_r2 + fp_r3.shape[1]
    end_mc = end_r3 + maccs.shape[1]
    feature_splits = np.array([end_r2, end_r3, end_mc], dtype=np.int32)
    return X, valid_mask, desc_mean, scaler, feature_splits


def main():
    print("[try4_preprocess] Loading data...")
    train_df = load_and_preprocess_data()
    print(f"  Train raw size: {len(train_df)}")

    # Robust target
    print("[try4_preprocess] Converting IC50 -> pIC50 with clipping...")
    train_df['pIC50'], bounds = robust_ic50_to_pic50(train_df['ic50_nM'].values)
    print(f"  log10(IC50) clip bounds: {bounds}")

    # Build features for training
    print("[try4_preprocess] Building train features...")
    X_train, valid_mask_train, desc_mean, scaler, splits = build_feature_matrix(
        train_df[['smiles']], fit=True
    )
    y_train = train_df.loc[valid_mask_train, 'pIC50'].values.astype(np.float32)

    print(f"  X_train shape: {X_train.shape}")

    # Save artifacts
    print("[try4_preprocess] Saving artifacts and preprocessed datasets...")
    feature_info = {
        'splits': {'end_morgan_r2': int(splits[0]), 'end_morgan_r3': int(splits[1]), 'end_maccs': int(splits[2])},
        'desc_mean_shape': int(len(desc_mean)),
        'bounds_log10': {'lower': bounds[0], 'upper': bounds[1]},
        'columns': {
            'X_train': 'X_train.csv',
            'X_test': 'X_test.csv',
            'y_train': 'y_train.csv',
            'test_ids': 'test_ids.csv'
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'feature_info.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_info, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump({'scaler': scaler, 'desc_mean': desc_mean}, f)

    # Persist train
    pd.DataFrame(X_train).to_csv(os.path.join(OUTPUT_DIR, 'X_train.csv'), index=False)
    pd.DataFrame({'pIC50': y_train}).to_csv(os.path.join(OUTPUT_DIR, 'y_train.csv'), index=False)

    # Process test
    print("[try4_preprocess] Processing test set...")
    test_df = pd.read_csv("./data/test.csv")
    test_df = test_df.rename(columns={'Smiles': 'smiles'})
    X_test, valid_mask_test, _, _, _ = build_feature_matrix(test_df[['smiles']], desc_mean=desc_mean, scaler=scaler, fit=False)
    print(f"  X_test(valid) shape: {X_test.shape}")

    # Save test matrices and valid IDs
    pd.DataFrame(X_test).to_csv(os.path.join(OUTPUT_DIR, 'X_test.csv'), index=False)
    pd.DataFrame({'ID': test_df.loc[valid_mask_test, 'ID'].values}).to_csv(
        os.path.join(OUTPUT_DIR, 'test_ids.csv'), index=False
    )

    # Also keep raw train/test info for convenience
    pd.DataFrame({'smiles': train_df.loc[valid_mask_train, 'smiles'].values, 'pIC50': y_train}).to_csv(
        os.path.join(OUTPUT_DIR, 'train_smiles_pic50.csv'), index=False
    )
    pd.DataFrame({'ID': test_df['ID'], 'smiles': test_df['smiles'], 'is_valid': valid_mask_test}).to_csv(
        os.path.join(OUTPUT_DIR, 'test_smiles_valid.csv'), index=False
    )

    print("[try4_preprocess] Done.")


if __name__ == "__main__":
    main()


