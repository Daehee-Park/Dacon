# try2.py: Stacking Ensemble with Hyperparameter Optimization
import pandas as pd
import numpy as np
import os
import random
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import r2_score, mean_squared_error
import optuna
import warnings
import gc

# 경고 메시지 무시
warnings.filterwarnings(action='ignore', message=".*DEPRECATION WARNING:.*")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings('ignore', category=UserWarning)
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# --- 설정 ---
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

OUTPUT_DIR = "./output/try2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 데이터 로딩 및 전처리 (baseline, try1 기반) ---
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

def preprocess_ic50_robust(ic50_values):
    """Capping과 안전한 로그 변환을 통한 pIC50 변환"""
    log_ic50 = np.log10(ic50_values + 1e-9)
    lower = np.percentile(log_ic50, 1)
    upper = np.percentile(log_ic50, 99)
    log_ic50_capped = np.clip(log_ic50, lower, upper)
    ic50_capped = 10 ** log_ic50_capped
    pic50 = 9 - np.log10(ic50_capped + 1e-9)
    return pic50

def pIC50_to_IC50(pIC50):
    return 10**(9 - pIC50)

# --- 피처 엔지니어링 (baseline 기반) ---
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=CFG['NBITS'])
        fp = morgan_gen.GetFingerprint(mol)
        arr = np.zeros((CFG['NBITS'],), dtype=np.int8)
        for i in range(CFG['NBITS']):
            arr[i] = fp.GetBit(i)
        return arr
    return None

def calculate_rdkit_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return np.full((len(Descriptors._descList),), np.nan)
    descriptors = [desc_func(mol) for _, desc_func in Descriptors._descList]
    return np.array(descriptors)

# --- 평가 함수 ---
def get_score(y_true_ic50, y_pred_ic50, y_true_pic50, y_pred_pic50):
    rmse = mean_squared_error(y_true_ic50, y_pred_ic50) ** 0.5
    nrmse = rmse / (np.max(y_true_ic50) - np.min(y_true_ic50))
    A = 1 - min(nrmse, 1)
    B = r2_score(y_true_pic50, y_pred_pic50)
    score = 0.4 * A + 0.6 * B
    return score

# --- 모델링 ---
def run_cv_and_predict(model, X, y, X_test, y_bins):
    """CV를 수행하여 OOF 예측과 테스트 예측을 반환"""
    skf = StratifiedKFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 모델에 따라 학습 방식 분기
        if isinstance(model, lgb.LGBMRegressor):
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      eval_metric='rmse', callbacks=[lgb.early_stopping(100, verbose=False)])
        elif isinstance(model, cb.CatBoostRegressor):
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      early_stopping_rounds=100, verbose=False)
        else: # ExtraTreesRegressor
            model.fit(X_train, y_train)

        oof_preds[val_idx] = model.predict(X_val)
        test_preds += model.predict(X_test) / CFG['N_SPLITS']
        
    y_ic50_true = pIC50_to_IC50(y)
    oof_ic50_preds = pIC50_to_IC50(oof_preds)
    cv_score = get_score(y_ic50_true, oof_ic50_preds, y, oof_preds)
    
    return oof_preds, test_preds, cv_score


# --- Optuna 목적 함수 ---
def create_objective(model_name):
    def objective(trial, X, y, y_bins):
        if model_name == 'lgb':
            params = {
                'objective': 'regression', 'metric': 'rmse', 'verbose': -1, 'n_jobs': CFG['CPUS'],
                'seed': CFG['SEED'], 'boosting_type': 'gbdt', 'n_estimators': 2000,
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            }
            model = lgb.LGBMRegressor(**params)
        elif model_name == 'cb':
            params = {
                'iterations': 2000, 'verbose': False, 'random_seed': CFG['SEED'], 'thread_count': CFG['CPUS'],
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            }
            model = cb.CatBoostRegressor(**params)
        elif model_name == 'et':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'n_jobs': CFG['CPUS'], 'random_state': CFG['SEED']
            }
            model = ExtraTreesRegressor(**params)

        _, _, score = run_cv_and_predict(model, X, y, X[:1], y_bins) # X_test는 사용 안함
        return score
    return objective

if __name__ == "__main__":
    print("--- Stacking Ensemble Pipeline Start ---")

    # 1. 데이터 로딩 및 전처리
    print("\n1. Loading and preprocessing data...")
    train_df = load_and_preprocess_data()
    train_df['pIC50'] = preprocess_ic50_robust(train_df['ic50_nM'])

    # 2. 피처 엔지니어링
    print("\n2. Feature engineering...")
    train_df['fingerprint'] = train_df['smiles'].apply(smiles_to_fingerprint)
    train_df['descriptors'] = train_df['smiles'].apply(calculate_rdkit_descriptors)
    train_df.dropna(subset=['fingerprint', 'descriptors'], inplace=True)

    desc_stack = np.stack(train_df['descriptors'].values)
    desc_mean = np.nanmean(desc_stack, axis=0)
    desc_stack = np.nan_to_num(desc_stack, nan=desc_mean)

    scaler = StandardScaler()
    desc_scaled = scaler.fit_transform(desc_stack)
    fp_stack = np.stack(train_df['fingerprint'].values)
    X = np.hstack([fp_stack, desc_scaled]).astype(np.float32)
    y = train_df['pIC50'].values
    y_bins = pd.qcut(y, q=CFG['N_SPLITS'], labels=False, duplicates='drop')
    print(f"  Feature matrix shape: {X.shape}")

    # 테스트 데이터 준비
    test_df = pd.read_csv("./data/test.csv")
    test_df['fingerprint'] = test_df['Smiles'].apply(smiles_to_fingerprint)
    test_df['descriptors'] = test_df['Smiles'].apply(calculate_rdkit_descriptors)
    valid_test_mask = test_df['fingerprint'].notna()
    
    fp_test_stack = np.stack(test_df.loc[valid_test_mask, 'fingerprint'].values)
    desc_test_stack = np.stack(test_df.loc[valid_test_mask, 'descriptors'].values)
    desc_test_stack = np.nan_to_num(desc_test_stack, nan=desc_mean)
    desc_test_scaled = scaler.transform(desc_test_stack)
    X_test = np.hstack([fp_test_stack, desc_test_scaled]).astype(np.float32)

    # 3. 1단계 모델 최적화 및 예측
    print("\n3. Training Level 1 Models...")
    base_models = ['lgb', 'cb', 'et']
    oof_preds_dict = {}
    test_preds_dict = {}

    for model_name in base_models:
        print(f"\n  Optimizing {model_name.upper()}...")
        objective_fn = create_objective(model_name)
        study = optuna.create_study(direction='maximize', study_name=f'{model_name}_tuning')
        study.optimize(lambda trial: objective_fn(trial, X, y, y_bins), n_trials=CFG['N_TRIALS'])
        
        best_params = study.best_params
        print(f"    Best CV score: {study.best_value:.4f}")

        # 최적 파라미터로 모델 생성 및 학습/예측
        if model_name == 'lgb':
            base_model = lgb.LGBMRegressor(objective='regression', metric='rmse', verbose=-1, n_jobs=CFG['CPUS'], seed=CFG['SEED'], boosting_type='gbdt', n_estimators=2000, **best_params)
        elif model_name == 'cb':
            base_model = cb.CatBoostRegressor(iterations=2000, verbose=False, random_seed=CFG['SEED'], thread_count=CFG['CPUS'], **best_params)
        elif model_name == 'et':
            base_model = ExtraTreesRegressor(n_jobs=CFG['CPUS'], random_state=CFG['SEED'], **best_params)
        
        print(f"  Training {model_name.upper()} with best params...")
        oof_preds, test_preds, cv_score = run_cv_and_predict(base_model, X, y, X_test, y_bins)
        oof_preds_dict[model_name] = oof_preds
        test_preds_dict[model_name] = test_preds
        print(f"    Final CV score for {model_name.upper()}: {cv_score:.4f}")
        gc.collect()

    # 4. 2단계 모델 (스태킹)
    print("\n4. Training Level 2 Meta-Model (Stacking)...")
    X_meta_train = np.column_stack(list(oof_preds_dict.values()))
    X_meta_test = np.column_stack(list(test_preds_dict.values()))

    meta_model = lgb.LGBMRegressor(random_state=CFG['SEED'], n_jobs=CFG['CPUS'])
    meta_model.fit(X_meta_train, y)
    
    # 최종 OOF 점수 계산
    meta_oof_preds = meta_model.predict(X_meta_train)
    meta_y_ic50_true = pIC50_to_IC50(y)
    meta_oof_ic50_preds = pIC50_to_IC50(meta_oof_preds)
    final_cv_score = get_score(meta_y_ic50_true, meta_oof_ic50_preds, y, meta_oof_preds)
    print(f"  Final Stacking CV Score: {final_cv_score:.4f}")

    final_test_preds = meta_model.predict(X_meta_test)
    final_ic50_preds = pIC50_to_IC50(final_test_preds)

    # 5. 제출 파일 생성
    print("\n5. Creating submission file...")
    submission_df = pd.read_csv("./data/sample_submission.csv")
    pred_df = pd.DataFrame({'ID': test_df.loc[valid_test_mask, 'ID'], 'ASK1_IC50_nM': final_ic50_preds})
    
    submission_df = submission_df[['ID']].merge(pred_df, on='ID', how='left')
    submission_df['ASK1_IC50_nM'].fillna(train_df['ic50_nM'].median(), inplace=True)
    
    submission_path = f"{OUTPUT_DIR}/submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"  Submission saved to {submission_path}")

    # 6. 제출
    try:
        from dacon_submit import dacon_submit
        dacon_submit(
            submission_path=submission_path,
            memo=f"try2: Stacking (LGB, CB, ET), CV {final_cv_score:.6f}"
        )
    except ImportError:
        print("  'dacon_submit' not found. Skipping submission.")

    print("\n--- Pipeline Complete ---") 