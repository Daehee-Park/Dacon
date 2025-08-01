# try1.py
import pandas as pd
import numpy as np
import os
import random
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, Lipinski, rdFingerprintGenerator
from rdkit.Chem.AtomPairs import Pairs, Torsions
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize
import optuna
import warnings
import gc

warnings.filterwarnings(action='ignore', message=".*DEPRECATION WARNING:.*")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings('ignore', category=UserWarning)
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL) 

CFG = {
    'NBITS': 2048,
    'SEED': 33,
    'N_SPLITS': 5,
    'N_TRIALS': 100,
    'CPUS': 64
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED'])

OUTPUT_DIR = "./output/try1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_preprocess_data():
    try:
        chembl = pd.read_csv("./data/ChEMBL_ASK1(IC50).csv", sep=';')
        pubchem = pd.read_csv("./data/Pubchem_ASK1.csv", low_memory=False)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure data files are in the current directory.")
        return None

    chembl.columns = chembl.columns.str.strip().str.replace('"', '')
    chembl = chembl[chembl['Standard Type'] == 'IC50']
    chembl = chembl[['Smiles', 'Standard Value']].rename(columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'})
    chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')

    pubchem = pubchem[['SMILES', 'Activity_Value']].rename(columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'})
    pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')

    df = pd.concat([chembl, pubchem], ignore_index=True).dropna(subset=['smiles', 'ic50_nM'])
    df = df.drop_duplicates(subset='smiles').reset_index(drop=True)
    df = df[df['ic50_nM'] > 0]

    return df

def extract_enhanced_features(smiles):
    """Extract multiple fingerprints and molecular descriptors"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = []
    
    # Morgan fingerprints with different radii
    morgan2_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=CFG['NBITS'])
    morgan2_fp = morgan2_gen.GetFingerprint(mol)
    morgan2_arr = np.zeros(CFG['NBITS'], dtype=np.int8)
    for i in range(CFG['NBITS']):
        morgan2_arr[i] = morgan2_fp.GetBit(i)
    features.extend(morgan2_arr)
    
    morgan3_gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024)
    morgan3_fp = morgan3_gen.GetFingerprint(mol)
    morgan3_arr = np.zeros(1024, dtype=np.int8)
    for i in range(1024):
        morgan3_arr[i] = morgan3_fp.GetBit(i)
    features.extend(morgan3_arr)
    
    # MACCS keys
    maccs = AllChem.GetMACCSKeysFingerprint(mol)
    maccs_arr = np.zeros(167, dtype=np.int8)
    for i in range(167):
        maccs_arr[i] = maccs.GetBit(i)
    features.extend(maccs_arr)
    
    # AtomPair fingerprint (fixed - handle properly)
    ap_fp = Pairs.GetAtomPairFingerprint(mol)
    # Convert to fixed-size array
    ap_arr = np.zeros(512, dtype=np.int8)
    for i, (key, value) in enumerate(ap_fp.GetNonzeroElements().items()):
        if i < 512:  # Limit to 512 features
            ap_arr[i] = min(value, 255)  # Cap values
    features.extend(ap_arr)
    
    # Topological Torsion fingerprint (fixed - handle properly)
    tt_fp = Torsions.GetTopologicalTorsionFingerprint(mol)
    # Convert to fixed-size array
    tt_arr = np.zeros(512, dtype=np.int8)
    for i, (key, value) in enumerate(tt_fp.GetNonzeroElements().items()):
        if i < 512:  # Limit to 512 features
            tt_arr[i] = min(value, 255)  # Cap values
    features.extend(tt_arr)
    
    # Key molecular descriptors
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.RingCount(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.BertzCT(mol),
        Descriptors.Chi0(mol),
        Descriptors.Chi1(mol),
        Descriptors.Kappa1(mol),
        Descriptors.Kappa2(mol),
        Descriptors.LabuteASA(mol),
        Descriptors.MaxEStateIndex(mol),
        Descriptors.MinEStateIndex(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumSaturatedHeterocycles(mol),
        Descriptors.NumAromaticHeterocycles(mol),
        Descriptors.MolMR(mol),
        Descriptors.qed(mol)
    ]
    
    features.extend(descriptors)
    
    return np.array(features, dtype=np.float32)

def preprocess_ic50_robust(ic50_values):
    """Robust IC50 preprocessing with outlier capping"""
    # Log transform first for better outlier detection
    log_ic50 = np.log10(ic50_values + 1e-9)
    
    # Cap outliers at 1st and 99th percentile
    lower = np.percentile(log_ic50, 1)
    upper = np.percentile(log_ic50, 99)
    log_ic50_capped = np.clip(log_ic50, lower, upper)
    
    # Convert back to IC50
    ic50_capped = 10 ** log_ic50_capped
    
    # Convert to pIC50
    pic50 = 9 - np.log10(ic50_capped + 1e-9)
    
    return pic50, (lower, upper)

def IC50_to_pIC50(ic50_nM): 
    return 9 - np.log10(ic50_nM + 1e-9)

def pIC50_to_IC50(pIC50): 
    return 10**(9 - pIC50)

def get_score(y_true_ic50, y_pred_ic50, y_true_pic50, y_pred_pic50):
    rmse = mean_squared_error(y_true_ic50, y_pred_ic50) ** 0.5
    nrmse = rmse / (np.max(y_true_ic50) - np.min(y_true_ic50))
    A = 1 - min(nrmse, 1)
    B = r2_score(y_true_pic50, y_pred_pic50)
    score = 0.4 * A + 0.6 * B
    return score

def train_model_cv(X, y, model_type='lgb', params=None):
    """Train a model with stratified k-fold CV"""
    # Create stratified folds based on pIC50 quantiles
    y_bins = pd.qcut(y, q=CFG['N_SPLITS'], labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
    
    oof_preds = np.zeros(len(X))
    models = []
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
        elif model_type == 'cb':
            model = cb.CatBoostRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=100,
                verbose=False
            )
        elif model_type == 'xgb':
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        
        oof_preds[val_idx] = model.predict(X_val)
        models.append(model)
        
        # Calculate fold score
        val_ic50_true = pIC50_to_IC50(y_val)
        val_ic50_pred = pIC50_to_IC50(oof_preds[val_idx])
        fold_score = get_score(val_ic50_true, val_ic50_pred, y_val, oof_preds[val_idx])
        scores.append(fold_score)
        
        print(f"  Fold {fold+1}/{CFG['N_SPLITS']} - Score: {fold_score:.4f}")
    
    # Overall CV score
    y_ic50_true = pIC50_to_IC50(y)
    oof_ic50_pred = pIC50_to_IC50(oof_preds)
    cv_score = get_score(y_ic50_true, oof_ic50_pred, y, oof_preds)
    
    print(f"  Overall CV Score: {cv_score:.4f} (±{np.std(scores):.4f})")
    
    return models, oof_preds, cv_score

def optimize_blend_weights(oof_dict, y):
    """Optimize ensemble weights to maximize CV score"""
    model_names = list(oof_dict.keys())
    n_models = len(model_names)
    
    def objective(weights):
        blended = np.zeros_like(y)
        for i, name in enumerate(model_names):
            blended += weights[i] * oof_dict[name]
        
        y_ic50_true = pIC50_to_IC50(y)
        blended_ic50 = pIC50_to_IC50(blended)
        score = get_score(y_ic50_true, blended_ic50, y, blended)
        return -score  # Minimize negative score
    
    # Initial weights (equal)
    x0 = np.ones(n_models) / n_models
    
    # Constraints: weights sum to 1, all non-negative
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = [(0, 1) for _ in range(n_models)]
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return dict(zip(model_names, result.x)), -result.fun

if __name__ == "__main__":
    print("=== Enhanced Drug Discovery Model ===")
    print("1. Loading and preprocessing data...")
    train_df = load_and_preprocess_data()
    
    if train_df is not None:
        # Robust IC50 preprocessing
        train_df['pIC50'], outlier_bounds = preprocess_ic50_robust(train_df['ic50_nM'])
        print(f"  IC50 outlier bounds (log10): [{outlier_bounds[0]:.2f}, {outlier_bounds[1]:.2f}]")
        
        print("\n2. Feature Engineering...")
        print("  Extracting enhanced molecular features...")
        train_df['features'] = train_df['smiles'].apply(extract_enhanced_features)
        train_df = train_df.dropna(subset=['features'])
        
        # Stack features and scale
        X = np.vstack(train_df['features'].values)
        y = train_df['pIC50'].values
        
        # Separate fingerprints and descriptors for scaling
        n_fingerprints = CFG['NBITS'] + 1024 + 167 + 512 + 512  # All fingerprint features
        X_fp = X[:, :n_fingerprints]
        X_desc = X[:, n_fingerprints:]
        
        # Scale descriptors only
        scaler = StandardScaler()
        X_desc_scaled = scaler.fit_transform(X_desc)
        X = np.hstack([X_fp, X_desc_scaled]).astype(np.float32)
        
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Target range (pIC50): [{y.min():.2f}, {y.max():.2f}]")
        
        print("\n3. Training Ensemble Models...")
        
        # Model configurations
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 50,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'n_estimators': 3000,
            'n_jobs': CFG['CPUS'],
            'seed': CFG['SEED'],
            'verbose': -1,
            'n_jobs': CFG['CPUS']
        }
        
        cb_params = {
            'iterations': 3000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': CFG['SEED'],
            'thread_count': CFG['CPUS'],
            'verbose': False,
            'n_jobs': CFG['CPUS']
        }
        
        xgb_params = {
            'n_estimators': 3000,
            'learning_rate': 0.03,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': CFG['SEED'],
            'n_jobs': CFG['CPUS'],
            'verbosity': 0,
            'n_jobs': CFG['CPUS']
        }
        
        # Train models
        all_models = {}
        oof_predictions = {}
        
        print("\n  Training LightGBM...")
        lgb_models, lgb_oof, lgb_score = train_model_cv(X, y, 'lgb', lgb_params)
        all_models['lgb'] = lgb_models
        oof_predictions['lgb'] = lgb_oof
        
        print("\n  Training CatBoost...")
        cb_models, cb_oof, cb_score = train_model_cv(X, y, 'cb', cb_params)
        all_models['cb'] = cb_models
        oof_predictions['cb'] = cb_oof
        
        print("\n  Training XGBoost...")
        xgb_models, xgb_oof, xgb_score = train_model_cv(X, y, 'xgb', xgb_params)
        all_models['xgb'] = xgb_models
        oof_predictions['xgb'] = xgb_oof
        
        # Optimize ensemble weights
        print("\n4. Optimizing Ensemble Weights...")
        optimal_weights, ensemble_score = optimize_blend_weights(oof_predictions, y)
        print(f"  Optimal weights: {optimal_weights}")
        print(f"  Ensemble CV Score: {ensemble_score:.4f}")
        
        # Test predictions
        print("\n5. Generating Test Predictions...")
        test_df = pd.read_csv("./data/test.csv")
        test_df['features'] = test_df['Smiles'].apply(extract_enhanced_features)
        
        # Handle invalid SMILES
        valid_mask = test_df['features'].notna()
        X_test_valid = np.vstack(test_df.loc[valid_mask, 'features'].values)
        
        # Scale test descriptors
        X_test_fp = X_test_valid[:, :n_fingerprints]
        X_test_desc = X_test_valid[:, n_fingerprints:]
        X_test_desc_scaled = scaler.transform(X_test_desc)
        X_test = np.hstack([X_test_fp, X_test_desc_scaled]).astype(np.float32)
        
        # Generate predictions for each model type
        test_preds = np.zeros(len(X_test))
        
        for model_name, weight in optimal_weights.items():
            model_preds = np.zeros(len(X_test))
            for model in all_models[model_name]:
                model_preds += model.predict(X_test) / len(all_models[model_name])
            test_preds += weight * model_preds
        
        # Convert to IC50
        test_ic50_preds = pIC50_to_IC50(test_preds)
        
        # Create submission
        print("\n6. Creating Submission File...")
        submission_df = pd.read_csv("./data/sample_submission.csv")
        
        # Map predictions to all test samples
        pred_df = pd.DataFrame({
            'ID': test_df.loc[valid_mask, 'ID'],
            'ASK1_IC50_nM': test_ic50_preds
        })
        
        submission_df = submission_df[['ID']].merge(pred_df, on='ID', how='left')
        
        # Fill invalid predictions with training mean
        submission_df['ASK1_IC50_nM'].fillna(train_df['ic50_nM'].median(), inplace=True)
        
        # Save submission
        submission_df.to_csv(f"{OUTPUT_DIR}/submission.csv", index=False)
        print(f"  Submission saved to {OUTPUT_DIR}/submission.csv")
        
        # Submit
        from dacon_submit import dacon_submit
        
        dacon_submit(
            submission_path=f"{OUTPUT_DIR}/submission.csv",
            memo=f"Enhanced ensemble (LGB+CB+XGB), CV {ensemble_score:.8f}"
        )
        
        print("\n=== Pipeline Complete ===")
        print(f"Final CV Score: {ensemble_score:.4f}")
        print(f"Models: {', '.join([f'{k}({v:.1%})' for k,v in optimal_weights.items()])}")