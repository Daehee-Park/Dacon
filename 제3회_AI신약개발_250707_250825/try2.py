import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Crippen
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from scipy.stats import pearsonr
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# 출력 디렉토리 생성
os.makedirs('output/try2', exist_ok=True)

# 로깅 설정
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

# 로그 파일 설정
log_filename = f'output/try2/print_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logger = Logger(log_filename)
sys.stdout = logger

CFG = {
    'NBITS': 2048,
    'SEED': 42,
    'N_FOLDS': 8,  # 8-fold CV
    'OPTUNA_TRIALS': 100,
    'MORGAN_RADIUS': 2,
    'REMOVE_OUTLIERS': True,
    'USE_MOLECULAR_FEATURES': True,
    'SCALE_FEATURES': True,
    'FEATURE_SELECTION': True,
    'N_FEATURES': 800,  # Feature 수 제한
    'ENSEMBLE_WEIGHTS': True
}

def seed_everything(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED'])

def save_config(config, filepath):
    """설정 저장"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(filepath):
    """설정 로드"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def save_optuna_params(params, filepath):
    """Optuna 최적화 결과 저장"""
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)

def load_optuna_params(filepath):
    """Optuna 최적화 결과 로드"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

# Try1과 동일한 전처리 함수들
def calculate_molecular_descriptors(smiles):
    """도메인 지식 기반 분자 특성 계산"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptors = {
        'MW': Descriptors.MolWt(mol),
        'LogP': Crippen.MolLogP(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'TPSA': Descriptors.TPSA(mol),
        'Rotatable': Descriptors.NumRotatableBonds(mol),
        'Aromatic_Rings': Descriptors.NumAromaticRings(mol),
        'Heavy_Atoms': Descriptors.HeavyAtomCount(mol),
        'Lipinski_Violations': int(Descriptors.NumHDonors(mol) > 5 or 
                                  Descriptors.NumHAcceptors(mol) > 10 or 
                                  Descriptors.MolWt(mol) > 500 or 
                                  Crippen.MolLogP(mol) > 5),
        'SlogP': Descriptors.SlogP_VSA1(mol),
        'LabuteASA': Descriptors.LabuteASA(mol),
        'BalabanJ': Descriptors.BalabanJ(mol),
        'BertzCT': Descriptors.BertzCT(mol),
        'FractionCsp3': Descriptors.FractionCSP3(mol),
        'RingCount': Descriptors.RingCount(mol),
        'MolMR': Crippen.MolMR(mol),
    }
    return descriptors

def smiles_to_fingerprint(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
        fp = morgan_gen.GetFingerprintAsNumPy(mol)
        return fp
    else:
        return np.zeros((nbits,))

def IC50_to_pIC50(ic50_nM):
    ic50_nM = np.clip(ic50_nM, 1e-3, None)
    return 9 - np.log10(ic50_nM)

def pIC50_to_IC50(pIC50):
    return 10 ** (9 - pIC50)

def detect_outliers_iqr(data, column, factor=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound)

def create_stratified_folds(y, molecular_features, n_splits=8):
    pIC50_bins = pd.cut(y, bins=[0, 5, 7, 9, 15], labels=['weak', 'moderate', 'strong', 'very_strong'])
    mw_bins = pd.cut(molecular_features['MW'], bins=4, labels=['small', 'medium', 'large', 'very_large'])
    combined_strata = pIC50_bins.astype(str) + '_' + mw_bins.astype(str)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG['SEED'])
    return skf.split(molecular_features, combined_strata)

def calculate_metrics(y_true_ic50, y_pred_ic50, y_true_pic50, y_pred_pic50):
    rmse = np.sqrt(mean_squared_error(y_true_ic50, y_pred_ic50))
    y_range = np.max(y_true_ic50) - np.min(y_true_ic50)
    normalized_rmse = rmse / y_range
    
    corr, _ = pearsonr(y_true_pic50, y_pred_pic50)
    correlation_squared = corr ** 2
    
    A = normalized_rmse
    B = correlation_squared
    final_score = 0.4 * (1 - min(A, 1)) + 0.6 * B
    
    return normalized_rmse, correlation_squared, final_score

# Optuna 최적화 함수들
def optimize_extratrees(X, y, cv_folds, save_path):
    """ExtraTrees 최적화 (RandomForest 대신)"""
    if os.path.exists(save_path):
        print(f"기존 최적화 결과 로드: {save_path}")
        return load_optuna_params(save_path)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.8]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': CFG['SEED']
        }
        
        scores = []
        for train_idx, val_idx in cv_folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = ExtraTreesRegressor(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(score)
        
        return np.mean(scores)
    
    try:
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=CFG['SEED']))
        study.optimize(objective, n_trials=CFG['OPTUNA_TRIALS'])
        best_params = study.best_params
        save_optuna_params(best_params, save_path)
        return best_params
    except Exception as e:
        print(f"ExtraTrees 최적화 중 오류 발생: {e}")
        default_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': CFG['SEED']
        }
        save_optuna_params(default_params, save_path)
        return default_params

def optimize_xgboost(X, y, cv_folds, save_path):
    if os.path.exists(save_path):
        print(f"기존 최적화 결과 로드: {save_path}")
        return load_optuna_params(save_path)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'early_stopping_rounds': 50,  # 생성자에 포함
            'random_state': CFG['SEED']
        }
        
        scores = []
        for train_idx, val_idx in cv_folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = model.predict(X_val)
            except Exception as e:
                # early_stopping_rounds 제거하고 재시도
                params_no_early = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
                model = xgb.XGBRegressor(**params_no_early)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = model.predict(X_val)
            
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(score)
        
        return np.mean(scores)
    
    try:
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=CFG['SEED']))
        study.optimize(objective, n_trials=CFG['OPTUNA_TRIALS'])
        best_params = study.best_params
        save_optuna_params(best_params, save_path)
        return best_params
    except Exception as e:
        print(f"XGBoost 최적화 중 오류 발생: {e}")
        default_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1,
            'reg_lambda': 1,
            'random_state': CFG['SEED']
        }
        save_optuna_params(default_params, save_path)
        return default_params

def optimize_lightgbm(X, y, cv_folds, save_path):
    if os.path.exists(save_path):
        print(f"기존 최적화 결과 로드: {save_path}")
        return load_optuna_params(save_path)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': CFG['SEED'],
            'verbosity': -1  # 모델 자체의 verbosity
        }
        
        scores = []
        for train_idx, val_idx in cv_folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     callbacks=[
                         lgb.early_stopping(50, verbose=False),  # early stopping 메시지 끄기
                         lgb.log_evaluation(0)  # 평가 로그 끄기
                     ])
            y_pred = model.predict(X_val)
            
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(score)
        
        return np.mean(scores)
    
    try:
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=CFG['SEED']))
        study.optimize(objective, n_trials=CFG['OPTUNA_TRIALS'])
        best_params = study.best_params
        save_optuna_params(best_params, save_path)
        return best_params
    except Exception as e:
        print(f"LightGBM 최적화 중 오류 발생: {e}")
        default_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1,
            'reg_lambda': 1,
            'random_state': CFG['SEED'],
            'verbosity': -1
        }
        save_optuna_params(default_params, save_path)
        return default_params

def optimize_catboost(X, y, cv_folds, save_path):
    if os.path.exists(save_path):
        print(f"기존 최적화 결과 로드: {save_path}")
        return load_optuna_params(save_path)
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'early_stopping_rounds': 50,
            'random_state': CFG['SEED'],
            'verbose': False
        }
        
        scores = []
        for train_idx, val_idx in cv_folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                model = cb.CatBoostRegressor(**params)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                y_pred = model.predict(X_val)
            except Exception as e:
                # early_stopping_rounds 제거하고 재시도
                params_no_early = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
                model = cb.CatBoostRegressor(**params_no_early)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                         early_stopping_rounds=50, verbose=False)
                y_pred = model.predict(X_val)
            
            score = np.sqrt(mean_squared_error(y_val, y_pred))
            scores.append(score)
        
        return np.mean(scores)
    
    try:
        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=CFG['SEED']))
        study.optimize(objective, n_trials=CFG['OPTUNA_TRIALS'])
        best_params = study.best_params
        save_optuna_params(best_params, save_path)
        return best_params
    except Exception as e:
        print(f"CatBoost 최적화 중 오류 발생: {e}")
        default_params = {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bylevel': 0.8,
            'l2_leaf_reg': 3,
            'random_state': CFG['SEED'],
            'verbose': False
        }
        save_optuna_params(default_params, save_path)
        return default_params

try:
    print("=" * 60)
    print("Try #2: ExtraTrees + XGBoost + LightGBM + CatBoost 앙상블")
    print("=" * 60)

    # 1. 데이터 로드 및 전처리
    print("\n1. 데이터 로드 및 전처리")
    print("-" * 50)

    # 데이터 로드
    chembl = pd.read_csv("data/ChEMBL_ASK1(IC50).csv", sep=';')
    pubchem = pd.read_csv("data/Pubchem_ASK1.csv", low_memory=False)

    chembl.columns = chembl.columns.str.strip().str.replace('"', '')
    chembl = chembl[chembl['Standard Type'] == 'IC50']
    chembl = chembl[['Smiles', 'Standard Value']].rename(columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'}).dropna()
    chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')
    chembl = chembl.dropna()[chembl['ic50_nM'] > 0]
    chembl['source'] = 'ChEMBL'

    pubchem = pubchem[['SMILES', 'Activity_Value']].rename(columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'}).dropna()
    pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')
    pubchem = pubchem.dropna()[pubchem['ic50_nM'] > 0]
    pubchem['source'] = 'PubChem'

    total = pd.concat([chembl, pubchem], ignore_index=True)
    total = total.drop_duplicates(subset='smiles')

    print(f"전처리 후 데이터: {len(total):,} 개")

    # 2. 분자 특성 계산
    print("\n2. 분자 특성 계산")
    print("-" * 50)

    molecular_descriptors = []
    valid_indices = []

    for idx, smiles in enumerate(total['smiles']):
        descriptors = calculate_molecular_descriptors(smiles)
        if descriptors is not None:
            molecular_descriptors.append(descriptors)
            valid_indices.append(idx)

    total = total.iloc[valid_indices].reset_index(drop=True)
    molecular_df = pd.DataFrame(molecular_descriptors)

    # 이상치 제거
    total['pIC50'] = IC50_to_pIC50(total['ic50_nM'])

    if CFG['REMOVE_OUTLIERS']:
        outliers_pic50 = detect_outliers_iqr(total, 'pIC50', factor=2.0)
        outliers_mw = detect_outliers_iqr(molecular_df, 'MW', factor=2.0)
        total_outliers = outliers_pic50 | outliers_mw
        
        total = total[~total_outliers].reset_index(drop=True)
        molecular_df = molecular_df[~total_outliers].reset_index(drop=True)

    print(f"최종 데이터: {len(total):,} 개")

    # 3. Feature Engineering
    print("\n3. Feature Engineering")
    print("-" * 50)

    # Morgan Fingerprint
    fingerprints = []
    for smiles in total['smiles']:
        fp = smiles_to_fingerprint(smiles, radius=CFG['MORGAN_RADIUS'], nbits=CFG['NBITS'])
        fingerprints.append(fp)

    X_fingerprint = np.array(fingerprints)

    # 분자 특성 결합
    scaler = RobustScaler()
    molecular_scaled = scaler.fit_transform(molecular_df)
    X_combined = np.concatenate([X_fingerprint, molecular_scaled], axis=1)

    print(f"전체 Feature 차원: {X_combined.shape}")

    # 4. Feature Selection
    if CFG['FEATURE_SELECTION']:
        print("\n4. Feature Selection")
        print("-" * 50)
        
        y_pic50 = total['pIC50'].values
        
        # Feature selection 결과 저장/로드
        selector_path = 'output/try2/feature_selector.pkl'
        if os.path.exists(selector_path):
            print("기존 Feature Selector 로드")
            with open(selector_path, 'rb') as f:
                selector = pickle.load(f)
            X_selected = selector.transform(X_combined)
        else:
            print("Feature Selection 수행")
            selector = SelectKBest(score_func=mutual_info_regression, k=CFG['N_FEATURES'])
            X_selected = selector.fit_transform(X_combined, y_pic50)
            with open(selector_path, 'wb') as f:
                pickle.dump(selector, f)
        
        selected_indices = selector.get_support(indices=True)
        print(f"Selected Features: {X_selected.shape[1]} / {X_combined.shape[1]}")
    else:
        X_selected = X_combined

    # 5. Cross-Validation 설정
    y_pic50 = total['pIC50'].values
    y_ic50 = total['ic50_nM'].values

    fold_generator = list(create_stratified_folds(y_pic50, molecular_df, CFG['N_FOLDS']))
    print(f"\n8-Fold Cross-Validation 설정 완료")

    # 6. 모델별 Optuna 최적화
    print("\n5. 모델별 하이퍼파라미터 최적화")
    print("-" * 50)

    models_config = {}

    print("ExtraTrees 최적화...")
    et_params = optimize_extratrees(X_selected, y_pic50, fold_generator,
                                   'output/try2/et_params.json')
    models_config['ExtraTrees'] = et_params
    print(f"ET 최적 파라미터: {et_params}")

    print("\nXGBoost 최적화...")
    xgb_params = optimize_xgboost(X_selected, y_pic50, fold_generator, 
                                 'output/try2/xgb_params.json')
    models_config['XGBoost'] = xgb_params
    print(f"XGB 최적 파라미터: {xgb_params}")

    print("\nLightGBM 최적화...")
    lgb_params = optimize_lightgbm(X_selected, y_pic50, fold_generator,
                                  'output/try2/lgb_params.json')
    models_config['LightGBM'] = lgb_params
    print(f"LGB 최적 파라미터: {lgb_params}")

    print("\nCatBoost 최적화...")
    cb_params = optimize_catboost(X_selected, y_pic50, fold_generator,
                                 'output/try2/cb_params.json')
    models_config['CatBoost'] = cb_params
    print(f"CB 최적 파라미터: {cb_params}")

    # 7. 모델별 Cross-Validation
    print("\n6. 모델별 Cross-Validation")
    print("-" * 50)

    model_results = {}
    model_predictions = {}

    for model_name, params in models_config.items():
        print(f"\n{model_name} CV 시작...")
        
        cv_scores = []
        all_preds = []
        all_true = []
        
        for fold, (train_idx, val_idx) in enumerate(fold_generator):
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train_pic50, y_val_pic50 = y_pic50[train_idx], y_pic50[val_idx]
            y_train_ic50, y_val_ic50 = y_ic50[train_idx], y_ic50[val_idx]
            
            # 모델 생성 및 훈련
            if model_name == 'ExtraTrees':
                model = ExtraTreesRegressor(**params)
                model.fit(X_train, y_train_pic50)
            elif model_name == 'XGBoost':
                try:
                    model = xgb.XGBRegressor(**params)
                    model.fit(X_train, y_train_pic50, eval_set=[(X_val, y_val_pic50)], verbose=False)
                except Exception as e:
                    # early_stopping_rounds 제거하고 재시도
                    params_no_early = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
                    model = xgb.XGBRegressor(**params_no_early)
                    model.fit(X_train, y_train_pic50, eval_set=[(X_val, y_val_pic50)], verbose=False)
            elif model_name == 'LightGBM':
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train_pic50, eval_set=[(X_val, y_val_pic50)], 
                         callbacks=[
                             lgb.early_stopping(50, verbose=False),  # early stopping 메시지 끄기
                             lgb.log_evaluation(0)  # 평가 로그 끄기
                         ])
            elif model_name == 'CatBoost':
                try:
                    model = cb.CatBoostRegressor(**params)
                    model.fit(X_train, y_train_pic50, eval_set=[(X_val, y_val_pic50)], verbose=False)
                except Exception as e:
                    # early_stopping_rounds 제거하고 재시도
                    params_no_early = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
                    model = cb.CatBoostRegressor(**params_no_early)
                    model.fit(X_train, y_train_pic50, eval_set=[(X_val, y_val_pic50)], 
                             early_stopping_rounds=50, verbose=False)
            
            # 예측
            y_val_pred_pic50 = model.predict(X_val)
            y_val_pred_ic50 = pIC50_to_IC50(y_val_pred_pic50)
            
            # 평가
            normalized_rmse, correlation_squared, final_score = calculate_metrics(
                y_val_ic50, y_val_pred_ic50, y_val_pic50, y_val_pred_pic50
            )
            
            cv_scores.append(final_score)
            all_preds.extend(y_val_pred_pic50)
            all_true.extend(y_val_pic50)
        
        model_results[model_name] = {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_scores': cv_scores
        }
        
        model_predictions[model_name] = {
            'predictions': np.array(all_preds),
            'true_values': np.array(all_true)
        }
        
        print(f"{model_name} CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # 8. 앙상블 모델
    print("\n7. 앙상블 모델 구성")
    print("-" * 50)

    # 단순 평균 앙상블
    ensemble_pred = np.mean([model_predictions[name]['predictions'] for name in models_config.keys()], axis=0)
    ensemble_true = model_predictions['ExtraTrees']['true_values']

    # 앙상블 성능 계산
    ensemble_ic50_pred = pIC50_to_IC50(ensemble_pred)
    ensemble_ic50_true = pIC50_to_IC50(ensemble_true)

    ensemble_rmse, ensemble_corr, ensemble_score = calculate_metrics(
        ensemble_ic50_true, ensemble_ic50_pred, ensemble_true, ensemble_pred
    )

    print(f"앙상블 CV Score: {ensemble_score:.4f}")

    # 9. 결과 요약
    print("\n" + "=" * 60)
    print("모델별 성능 비교")
    print("=" * 60)

    results_summary = []
    for model_name, results in model_results.items():
        results_summary.append({
            'Model': model_name,
            'CV_Score': results['cv_mean'],
            'CV_Std': results['cv_std']
        })

    results_summary.append({
        'Model': 'Ensemble',
        'CV_Score': ensemble_score,
        'CV_Std': 0.0
    })

    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values('CV_Score', ascending=False)
    print(results_df)

    # 최고 성능 모델
    best_model_name = results_df.iloc[0]['Model']
    best_score = results_df.iloc[0]['CV_Score']

    print(f"\n최고 성능 모델: {best_model_name} (CV Score: {best_score:.4f})")
    print(f"Try #1 대비: {best_score - 0.5573:+.4f} ({(best_score - 0.5573) / 0.5573 * 100:+.1f}%)")

    # 10. 최종 모델 훈련 및 제출 파일 생성
    print("\n8. 최종 모델 훈련 및 제출")
    print("-" * 50)

    # 테스트 데이터 처리
    test_df = pd.read_csv("data/test.csv")

    test_molecular_descriptors = []
    test_fingerprints = []
    valid_test_indices = []

    for idx, smiles in enumerate(test_df['smiles']):
        descriptors = calculate_molecular_descriptors(smiles)
        fp = smiles_to_fingerprint(smiles, radius=CFG['MORGAN_RADIUS'], nbits=CFG['NBITS'])
        
        if descriptors is not None:
            test_molecular_descriptors.append(descriptors)
            test_fingerprints.append(fp)
            valid_test_indices.append(idx)

    test_df_valid = test_df.iloc[valid_test_indices].reset_index(drop=True)
    test_molecular_df = pd.DataFrame(test_molecular_descriptors)
    test_fingerprint_array = np.array(test_fingerprints)

    # 테스트 특성 결합
    test_molecular_scaled = scaler.transform(test_molecular_df)
    X_test_combined = np.concatenate([test_fingerprint_array, test_molecular_scaled], axis=1)

    if CFG['FEATURE_SELECTION']:
        X_test_selected = selector.transform(X_test_combined)
    else:
        X_test_selected = X_test_combined

    # 최종 모델들 훈련
    final_models = {}
    test_predictions = {}

    for model_name, params in models_config.items():
        print(f"{model_name} 최종 모델 훈련...")
        
        if model_name == 'ExtraTrees':
            model = ExtraTreesRegressor(**params)
        elif model_name == 'XGBoost':
            try:
                model = xgb.XGBRegressor(**params)
            except Exception as e:
                # early_stopping_rounds 제거하고 재시도
                params_no_early = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
                model = xgb.XGBRegressor(**params_no_early)
        elif model_name == 'LightGBM':
            model = lgb.LGBMRegressor(**params)
        elif model_name == 'CatBoost':
            try:
                model = cb.CatBoostRegressor(**params)
            except Exception as e:
                # early_stopping_rounds 제거하고 재시도
                params_no_early = {k: v for k, v in params.items() if k != 'early_stopping_rounds'}
                model = cb.CatBoostRegressor(**params_no_early)
        
        model.fit(X_selected, y_pic50)
        final_models[model_name] = model
        
        # 테스트 예측
        test_pred_pic50 = model.predict(X_test_selected)
        test_predictions[model_name] = test_pred_pic50

    # 앙상블 예측
    ensemble_test_pred = np.mean(list(test_predictions.values()), axis=0)
    ensemble_test_ic50 = pIC50_to_IC50(ensemble_test_pred)

    # 제출 파일 생성
    submission = pd.DataFrame({
        'ID': test_df_valid['ID'],
        'ASK1_IC50_nM': ensemble_test_ic50
    })

    submission.to_csv('output/try2/submission.csv', index=False)

    # 11. 결과 저장
    print("\n9. 결과 저장")
    print("-" * 50)

    # 결과 저장
    results_df.to_csv('output/try2/model_comparison.csv', index=False)
    save_config(CFG, 'output/try2/config.json')

    # 모델별 설정 저장
    with open('output/try2/all_model_configs.json', 'w') as f:
        json.dump(models_config, f, indent=2)

    print(f"결과 저장 완료:")
    print(f"- 모델 비교: output/try2/model_comparison.csv")
    print(f"- 제출파일: output/try2/submission.csv") 
    print(f"- 모델 설정: output/try2/all_model_configs.json")
    print(f"- 전체 설정: output/try2/config.json")
    print(f"- Feature Selector: output/try2/feature_selector.pkl")

    print(f"\n최종 앙상블 CV Score: {ensemble_score:.4f}")

except Exception as e:
    print(f"\n심각한 오류 발생: {e}")
    import traceback
    print(traceback.format_exc())
    
finally:
    # 로그 파일 정리
    sys.stdout = logger.terminal
    logger.close()
    print(f"로그가 저장되었습니다: {log_filename}") 