import pandas as pd
import numpy as np
import os
import random
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
import optuna
import warnings
from rdkit.Chem import rdFingerprintGenerator

warnings.filterwarnings(action='ignore', message=".*please use MorganGenerator.*")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

CFG = {
    'NBITS': 2048,
    'SEED': 33,
    'N_SPLITS': 5,
    'N_TRIALS': 200,
    'CPUS': 64
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED'])

OUTPUT_DIR = "./output/baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)  # 경로 없으면 생성

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

def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=CFG['NBITS'])
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((CFG['NBITS'],), dtype=int)
        for i in range(CFG['NBITS']):
            arr[i] = fp.GetBit(i)
        return arr
    return None

def calculate_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.full((len(Descriptors._descList),), np.nan)
    descriptors = [desc_func(mol) for _, desc_func in Descriptors._descList]
    return np.array(descriptors)

def IC50_to_pIC50(ic50_nM): return 9 - np.log10(ic50_nM)
def pIC50_to_IC50(pIC50): return 10**(9 - pIC50)

def get_score(y_true_ic50, y_pred_ic50, y_true_pic50, y_pred_pic50):
    rmse = mean_squared_error(y_true_ic50, y_pred_ic50)
    nrmse = rmse / (np.max(y_true_ic50) - np.min(y_true_ic50))
    A = 1 - min(nrmse, 1)
    B = r2_score(y_true_pic50, y_pred_pic50)
    score = 0.4 * A + 0.6 * B
    return score

def objective(trial, X, y):
    params = {
        'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'n_jobs': CFG['CPUS'],
        'seed': CFG['SEED'], 'boosting_type': 'gbdt', 'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
    }

    kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_preds[val_idx] = model.predict(X_val)

    y_ic50_true = pIC50_to_IC50(y)
    oof_ic50_preds = pIC50_to_IC50(oof_preds)
    score = get_score(y_ic50_true, oof_ic50_preds, y, oof_preds)
    return score

if __name__ == "__main__":
    print("1. Loading and preprocessing data...")
    train_df = load_and_preprocess_data()

    if train_df is not None:
        train_df['pIC50'] = IC50_to_pIC50(train_df['ic50_nM'])
        print("\n--- Feature Engineering ---")
        train_df['fingerprint'] = train_df['smiles'].apply(smiles_to_fingerprint)
        train_df['descriptors'] = train_df['smiles'].apply(calculate_rdkit_descriptors)
        train_df.dropna(subset=['fingerprint', 'descriptors'], inplace=True)

        desc_stack = np.stack(train_df['descriptors'].values)
        desc_mean = np.nanmean(desc_stack, axis=0)
        desc_stack = np.nan_to_num(desc_stack, nan=desc_mean)

        scaler = StandardScaler()
        desc_scaled = scaler.fit_transform(desc_stack)
        fp_stack = np.stack(train_df['fingerprint'].values)
        X = np.hstack([fp_stack, desc_scaled])
        y = train_df['pIC50'].values

        print("\n--- Starting Hyperparameter Optimization with Optuna ---")
        study = optuna.create_study(direction='maximize', study_name='lgbm_tuning')
        study.optimize(lambda trial: objective(trial, X, y), n_trials=CFG['N_TRIALS'], n_jobs=CFG['CPUS'])

        print(f"\nOptimization Finished. Best Score: {study.best_value:.4f}")
        print("Best Parameters:", study.best_params)

        best_params = { 'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'n_jobs':CFG['CPUS'],
                        'seed': CFG['SEED'], 'boosting_type': 'gbdt', 'n_estimators': 2000 }
        best_params.update(study.best_params)

        print("\n--- Training Final Model with Best Parameters ---")
        test_df = pd.read_csv("./data/test.csv")
        test_df['fingerprint'] = test_df['Smiles'].apply(smiles_to_fingerprint)
        test_df['descriptors'] = test_df['Smiles'].apply(calculate_rdkit_descriptors)

        valid_test_mask = test_df['fingerprint'].notna() & test_df['descriptors'].notna()
        fp_test_stack = np.stack(test_df.loc[valid_test_mask, 'fingerprint'].values)
        desc_test_stack = np.stack(test_df.loc[valid_test_mask, 'descriptors'].values)
        desc_test_stack = np.nan_to_num(desc_test_stack, nan=desc_mean)
        desc_test_scaled = scaler.transform(desc_test_stack)
        X_test = np.hstack([fp_test_stack, desc_test_scaled])

        kf = KFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
        test_preds = np.zeros(len(X_test))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"--- Training Fold {fold+1}/{CFG['N_SPLITS']} ---")
            X_train, y_train = X[train_idx], y[train_idx]
            model = lgb.LGBMRegressor(**best_params)
            model.fit(X_train, y_train)
            test_preds += model.predict(X_test) / CFG['N_SPLITS']

        print("\n3. Generating submission file...")
        submission_df = pd.read_csv("./data/sample_submission.csv")
        pred_df = pd.DataFrame({'ID': test_df.loc[valid_test_mask, 'ID'], 'ASK1_IC50_nM': pIC50_to_IC50(test_preds)})
        submission_df = submission_df[['ID']].merge(pred_df, on='ID', how='left')
        submission_df['ASK1_IC50_nM'].fillna(train_df['ic50_nM'].mean(), inplace=True)
        submission_df.to_csv(f"{OUTPUT_DIR}/submission.csv", index=False)
        print("Submission file")

        # ------------------------- Submit ------------------------------------------
        from dacon_submit import dacon_submit

        dacon_submit(
            submission_path=f"{OUTPUT_DIR}/submission.csv",
            memo=f"Baseline, CV {study.best_value:.8f}"
        ) 