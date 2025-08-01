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
from autogluon.tabular import TabularPredictor
from custom_metrics import competition_scorer

warnings.filterwarnings(action='ignore', message=".*please use MorganGenerator.*")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

CFG = {
    'NBITS': 256,
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

OUTPUT_DIR = "./output/try3"
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

if __name__ == "__main__":
    print("1. Loading and preprocessing data...")
    train_df = load_and_preprocess_data()

    if train_df is not None:
        train_df['pIC50'] = IC50_to_pIC50(train_df['ic50_nM'])
        print("\n--- Feature Engineering for Training Data ---")
        train_df['fingerprint'] = train_df['smiles'].apply(smiles_to_fingerprint)
        train_df['descriptors'] = train_df['smiles'].apply(calculate_rdkit_descriptors)
        train_df.dropna(subset=['fingerprint', 'descriptors'], inplace=True)

        # 학습 데이터 피처 생성
        desc_stack = np.stack(train_df['descriptors'].values)
        desc_mean = np.nanmean(desc_stack, axis=0)
        desc_stack = np.nan_to_num(desc_stack, nan=desc_mean)

        scaler = StandardScaler()
        desc_scaled = scaler.fit_transform(desc_stack)
        fp_stack = np.stack(train_df['fingerprint'].values)
        X_train = np.hstack([fp_stack, desc_scaled])
        y_train = train_df['pIC50'].values

        # DataFrame으로 변환 (AutoGluon용)
        feature_names = [f'fp_{i}' for i in range(CFG['NBITS'])] + [f'desc_{i}' for i in range(desc_scaled.shape[1])]
        train_data = pd.DataFrame(X_train, columns=feature_names)
        train_data['pIC50'] = y_train

        print("\n2. Training AutoGluon model...")
        predictor = TabularPredictor(
            label='pIC50', 
            eval_metric=competition_scorer,
            path=f"{OUTPUT_DIR}/autogluon_models"
        )
        predictor.fit(
            train_data=train_data, 
            time_limit=3600*8,
            presets='extreme_quality',
            num_cpus=CFG['CPUS'],
            memory_limit=256
        )
        
        print("\n3. Processing test data...")
        test_df = pd.read_csv("./data/test.csv")
        print(f"Test data shape: {test_df.shape}")
        
        # 테스트 데이터 피처 생성
        test_df['fingerprint'] = test_df['SMILES'].apply(smiles_to_fingerprint)
        test_df['descriptors'] = test_df['SMILES'].apply(calculate_rdkit_descriptors)
        
        # 유효한 테스트 데이터 마스크 생성
        valid_test_mask = test_df['fingerprint'].notna() & test_df['descriptors'].notna()
        valid_test_df = test_df[valid_test_mask].copy()
        
        if len(valid_test_df) > 0:
            test_desc_stack = np.stack(valid_test_df['descriptors'].values)
            test_desc_stack = np.nan_to_num(test_desc_stack, nan=desc_mean)
            test_desc_scaled = scaler.transform(test_desc_stack)
            test_fp_stack = np.stack(valid_test_df['fingerprint'].values)
            X_test = np.hstack([test_fp_stack, test_desc_scaled])
            
            # DataFrame으로 변환
            test_data = pd.DataFrame(X_test, columns=feature_names)
            
            # 예측 수행
            test_preds_pic50 = predictor.predict(test_data)
            test_preds_ic50 = pIC50_to_IC50(test_preds_pic50)
            
            print(f"Valid predictions: {len(test_preds_ic50)}")
        else:
            print("No valid test predictions available")
            test_preds_ic50 = []

        print("\n4. Generating submission file...")
        submission_df = pd.read_csv("./data/sample_submission.csv")
        
        if len(test_preds_ic50) > 0:
            pred_df = pd.DataFrame({
                'ID': valid_test_df['ID'].values, 
                'ASK1_IC50_nM': test_preds_ic50
            })
            submission_df = submission_df[['ID']].merge(pred_df, on='ID', how='left')
        
        # 예측값이 없는 경우 평균값으로 채움
        submission_df['ASK1_IC50_nM'].fillna(train_df['ic50_nM'].mean(), inplace=True)
        submission_df.to_csv(f"{OUTPUT_DIR}/submission.csv", index=False)
        
        print(f"Submission file saved to {OUTPUT_DIR}/submission.csv")
        print(f"Submission shape: {submission_df.shape}")
        
        # 모델 리더보드 저장
        leaderboard = predictor.leaderboard(silent=True)
        leaderboard.to_csv(f"{OUTPUT_DIR}/leaderboard.csv", index=False)
        print(f"Model leaderboard saved to {OUTPUT_DIR}/leaderboard.csv")
        
    else:
        print("Failed to load training data")