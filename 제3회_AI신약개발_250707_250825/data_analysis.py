import os
import json
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator


CFG: Dict[str, Any] = {
    'SEED': 33,
    'NBITS_LIST': [256, 512, 1024, 2048],
    'DESC_SAMPLE_N': 3000,     # descriptor/fp sampling size for speed
    'OUTPUT_DIR': os.path.join(os.path.dirname(__file__), 'output', 'try4'),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(CFG['SEED'])
os.makedirs(CFG['OUTPUT_DIR'], exist_ok=True)


# ------------------------------ SMILES utils ------------------------------
def canonicalize_smiles(raw_smiles: str) -> str:
    if not isinstance(raw_smiles, str) or len(raw_smiles) == 0:
        return None
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
    return Chem.MolToSmiles(best_mol, canonical=True)


def smiles_to_fingerprint(smiles: str, nbits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)
    fp = morgan_gen.GetFingerprint(mol)
    arr = np.zeros((nbits,), dtype=np.int32)
    for i in range(nbits):
        arr[i] = fp.GetBit(i)
    return arr


def calculate_rdkit_descriptors(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full((len(Descriptors._descList),), np.nan)
    desc_vals = [desc_func(mol) for _, desc_func in Descriptors._descList]
    return np.asarray(desc_vals, dtype=float)


# ------------------------------ Loaders -----------------------------------
def load_sources() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_dir = os.path.dirname(__file__)
    chembl = pd.read_csv(os.path.join(base_dir, 'data', 'ChEMBL_ASK1(IC50).csv'), sep=';')
    pubchem = pd.read_csv(os.path.join(base_dir, 'data', 'Pubchem_ASK1.csv'), low_memory=False)
    test = pd.read_csv(os.path.join(base_dir, 'data', 'test.csv'))
    return chembl, pubchem, test


def unify_training(chembl: pd.DataFrame, pubchem: pd.DataFrame) -> pd.DataFrame:
    chembl = chembl.copy()
    chembl.columns = chembl.columns.str.strip().str.replace('"', '')
    # Focus on IC50
    if 'Standard Type' in chembl.columns:
        chembl = chembl[chembl['Standard Type'] == 'IC50']
    # Value + Units
    if 'Standard Units' in chembl.columns:
        chembl['units'] = chembl['Standard Units']
        chembl['value'] = pd.to_numeric(chembl['Standard Value'], errors='coerce')
    else:
        chembl['units'] = np.nan
        chembl['value'] = pd.to_numeric(chembl['Standard Value'], errors='coerce')
    chembl = chembl.rename(columns={'Smiles': 'smiles'})[['smiles', 'value', 'units']]

    pub = pubchem.rename(columns={'SMILES': 'smiles', 'Activity_Value': 'value'})[['smiles', 'value']]
    pub['units'] = np.nan
    pub['value'] = pd.to_numeric(pub['value'], errors='coerce')

    df = pd.concat([chembl, pub], ignore_index=True)
    return df


# ------------------------------ Analyses ----------------------------------
def analyze_schema(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    info = {
        'name': name,
        'shape': list(df.shape),
        'columns': df.columns.tolist(),
        'dtypes': {c: str(t) for c, t in df.dtypes.items()},
        'num_missing': {c: int(df[c].isna().sum()) for c in df.columns},
        'head': df.head(3).to_dict(orient='list'),
    }
    return info


def summarize_units(train_df: pd.DataFrame) -> Dict[str, Any]:
    unit_summary = {}
    if 'units' in train_df.columns:
        counts = train_df['units'].fillna('NA').value_counts(dropna=False).to_dict()
        unit_summary['unit_counts'] = counts
        by_unit = train_df.groupby(train_df['units'].fillna('NA'))['value'].describe().reset_index()
        unit_summary['by_unit_describe'] = by_unit.to_dict(orient='list')
    return unit_summary


def ic50_nM_from_value_unit(value: float, unit: str) -> float:
    try:
        v = float(value)
    except Exception:
        return np.nan
    if unit is None or (isinstance(unit, float) and np.isnan(unit)):
        return v
    u = str(unit).strip().lower()
    if u in ['nm', 'nanomolar', 'nanomolars']:
        return v
    if u in ['um', 'µm', 'micromolar', 'micromolars']:
        return v * 1e3
    if u in ['mm', 'millimolar', 'millimolars']:
        return v * 1e6
    if u in ['pm', 'picomolar', 'picomolars']:
        return v * 1e-3
    return v


def analyze_values_and_smiles(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    # Canonicalize and salt removal
    df['smiles_canonical'] = df['smiles'].astype(str).map(canonicalize_smiles)
    df['has_salt'] = df['smiles'].astype(str).str.contains('\.')

    # Convert IC50 to nM using units when available
    df['ic50_nM'] = [ic50_nM_from_value_unit(v, u) for v, u in zip(df['value'], df.get('units', np.nan))]
    df = df.dropna(subset=['smiles_canonical', 'ic50_nM'])
    df = df[df['ic50_nM'] > 0]

    # Deduplicate by canonical SMILES (median IC50)
    agg = df.groupby('smiles_canonical', as_index=False).agg(
        ic50_nM=('ic50_nM', 'median'),
        n_records=('ic50_nM', 'size'),
        had_salt=('has_salt', 'any'),
    )

    # Basic stats
    stats = {
        'n_raw': int(len(df)),
        'n_unique_smiles': int(len(agg)),
        'salt_fraction_raw': float(df['has_salt'].mean()) if len(df) else 0.0,
        'multi_record_fraction': float((agg['n_records'] > 1).mean()) if len(agg) else 0.0,
    }

    # Distributions
    ic50 = agg['ic50_nM']
    pic50 = 9 - np.log10(ic50)
    dist = {
        'ic50_nM': {
            'min': float(np.min(ic50)),
            'p05': float(np.percentile(ic50, 5)),
            'p50': float(np.percentile(ic50, 50)),
            'p95': float(np.percentile(ic50, 95)),
            'max': float(np.max(ic50)),
        },
        'pIC50': {
            'min': float(np.min(pic50)),
            'p05': float(np.percentile(pic50, 5)),
            'p50': float(np.percentile(pic50, 50)),
            'p95': float(np.percentile(pic50, 95)),
            'max': float(np.max(pic50)),
        },
    }

    return {
        'stats': stats,
        'distributions': dist,
        'train_canonical': agg,
    }


def analyze_overlap(train_can: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    test = test_df.copy()
    test['smiles_canonical'] = test['Smiles'].astype(str).map(canonicalize_smiles)
    test_can = test.dropna(subset=['smiles_canonical'])
    train_set = set(train_can['smiles_canonical'])
    test_set = set(test_can['smiles_canonical'])
    inter = train_set.intersection(test_set)
    return {
        'n_test_valid': int(len(test_can)),
        'n_overlap': int(len(inter)),
        'overlap_fraction_test': float(len(inter) / max(1, len(test_set))),
    }


def analyze_descriptors_and_fps(train_can: pd.DataFrame) -> Dict[str, Any]:
    # Sample for speed
    df = train_can.copy()
    n = min(CFG['DESC_SAMPLE_N'], len(df))
    if n == 0:
        return {}
    samp = df.sample(n=n, random_state=CFG['SEED']).reset_index(drop=True)

    # Descriptors
    desc_vals = np.stack([calculate_rdkit_descriptors(s) for s in samp['smiles_canonical']])
    nan_rate = np.mean(np.isnan(desc_vals), axis=0)
    means = np.nanmean(desc_vals, axis=0)
    stds = np.nanstd(desc_vals, axis=0)

    desc_names = [name for name, _ in Descriptors._descList]
    desc_summary = pd.DataFrame({
        'descriptor': desc_names,
        'nan_rate': nan_rate,
        'mean': means,
        'std': stds,
    })

    # Correlation (after simple mean-impute)
    imp = np.where(np.isnan(desc_vals), means, desc_vals)
    if imp.shape[0] > 1:
        corr = np.corrcoef(imp, rowvar=False)
        # count highly correlated pairs
        triu = np.triu(np.ones_like(corr, dtype=bool), k=1)
        high_corr_pairs = int(np.sum(np.abs(corr[triu]) > 0.98))
    else:
        high_corr_pairs = 0

    # Fingerprint bit densities across candidate nbits
    fp_density: List[Dict[str, Any]] = []
    for nbits in CFG['NBITS_LIST']:
        fps = [smiles_to_fingerprint(s, nbits) for s in samp['smiles_canonical']]
        fps = np.stack([f for f in fps if f is not None])
        on_bits_per_mol = fps.sum(axis=1)
        density = float(np.mean(on_bits_per_mol) / nbits)
        fp_density.append({'nbits': nbits, 'mean_on_bits': float(np.mean(on_bits_per_mol)), 'density': density})

    # Save detailed descriptor summary
    desc_summary.to_csv(os.path.join(CFG['OUTPUT_DIR'], 'descriptor_summary.csv'), index=False)

    return {
        'descriptor_nan_rate_mean': float(desc_summary['nan_rate'].mean()),
        'descriptor_nan_rate_median': float(desc_summary['nan_rate'].median()),
        'descriptor_zero_std_fraction': float((desc_summary['std'] == 0).mean()),
        'high_corr_pairs_count_thr0_98': high_corr_pairs,
        'fp_density': fp_density,
    }


def recommend_preprocessing(summary: Dict[str, Any]) -> Dict[str, Any]:
    # Heuristic recommendation based on analysis
    fp_info = summary.get('descriptors_fps', {}).get('fp_density', [])
    # pick nbits yielding density around 0.02-0.10; fallback 1024
    best_nbits = 1024
    best_gap = 1.0
    for item in fp_info:
        gap = abs(item['density'] - 0.05)
        if gap < best_gap:
            best_gap = gap
            best_nbits = int(item['nbits'])

    rec = {
        'salt_removal': True,
        'canonicalize': True,
        'duplicate_aggregation': 'median',
        'unit_handling': 'convert_to_nM_if_available_else_as_is',
        'target_transform': 'pIC50 = 9 - log10(nM)',
        'descriptor_imputation': 'mean_by_column',
        'descriptor_scaler': 'StandardScaler',
        'descriptor_variance_threshold': 1e-10,
        'descriptor_corr_threshold': 0.98,
        'fingerprint': {
            'type': 'Morgan(radius=2)',
            'nbits': best_nbits,
        },
    }
    return rec


def main() -> None:
    chembl, pubchem, test = load_sources()

    # Schema summaries
    schema = {
        'chembl': analyze_schema(chembl, 'chembl'),
        'pubchem': analyze_schema(pubchem, 'pubchem'),
        'test': analyze_schema(test, 'test'),
    }

    # Unify training and analyze
    train_raw = unify_training(chembl, pubchem)
    unit_info = summarize_units(train_raw)
    values_smiles_info = analyze_values_and_smiles(train_raw)

    # Overlap with test
    overlap = analyze_overlap(values_smiles_info['train_canonical'], test)

    # Descriptor & fingerprint probes
    desc_fp = analyze_descriptors_and_fps(values_smiles_info['train_canonical'])

    # Recommendations
    summary = {
        'schema': schema,
        'units': unit_info,
        'values_smiles': {
            'stats': values_smiles_info['stats'],
            'distributions': values_smiles_info['distributions'],
        },
        'overlap_train_test': overlap,
        'descriptors_fps': desc_fp,
    }
    rec = recommend_preprocessing(summary)

    # Save analysis snapshot JSON
    with open(os.path.join(CFG['OUTPUT_DIR'], 'data_analysis_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save recommendations
    with open(os.path.join(CFG['OUTPUT_DIR'], 'preprocess_recommendations.json'), 'w', encoding='utf-8') as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    # Save canonical training table (lightweight snapshot)
    values_smiles_info['train_canonical'].to_csv(
        os.path.join(CFG['OUTPUT_DIR'], 'train_canonical_summary.csv'), index=False
    )

    # Human-readable text report
    lines: List[str] = []
    lines.append('# Data Analysis Report (try4)')
    lines.append('')
    lines.append('## Schema')
    for k in ['chembl', 'pubchem', 'test']:
        s = schema[k]
        lines.append(f"- {k} shape: {s['shape']}")
        lines.append(f"  columns: {s['columns']}")
    lines.append('')
    lines.append('## Units (Chembl/Pubchem unified)')
    lines.append(f"- unit_counts: {unit_info.get('unit_counts', {})}")
    lines.append('')
    lines.append('## Values & SMILES')
    lines.append(f"- stats: {values_smiles_info['stats']}")
    lines.append(f"- distributions (ic50_nM & pIC50 percentiles): {values_smiles_info['distributions']}")
    lines.append('')
    lines.append('## Train/Test Overlap (canonical)')
    lines.append(str(overlap))
    lines.append('')
    lines.append('## Descriptors/Fingerprints')
    lines.append(str(desc_fp))
    lines.append('')
    lines.append('## Recommended Preprocessing')
    lines.append(json.dumps(rec, ensure_ascii=False))

    with open(os.path.join(CFG['OUTPUT_DIR'], 'data_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('Analysis complete. Outputs saved under:', CFG['OUTPUT_DIR'])


if __name__ == '__main__':
    main()


