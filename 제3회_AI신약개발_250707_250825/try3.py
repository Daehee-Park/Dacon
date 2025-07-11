import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Crippen, rdMolDescriptors
from sklearn.model_selection import StratifiedKFold, GroupKFold
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
os.makedirs('output/try3', exist_ok=True)

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
log_filename = f'output/try3/print_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logger = Logger(log_filename)
sys.stdout = logger

CFG = {
    'NBITS': 2048,
    'SEED': 42,
    'N_FOLDS': 8,
    'OPTUNA_TRIALS': 120,  # 더 많은 시행
    'MORGAN_RADIUS': 2,
    'REMOVE_OUTLIERS': True,
    'USE_MOLECULAR_FEATURES': True,
    'SCALE_FEATURES': True,
    'FEATURE_SELECTION': True,
    'N_FEATURES': 800,
    'DOMAIN_ADAPTATION': True,
    'SAMPLE_WEIGHTING': True,
    'GROUP_CV': True,
    'ADVANCED_FEATURES': True
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

def save_model(model, filepath):
    """모델 저장"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    """모델 로드"""
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def calculate_molecular_descriptors_extended(smiles):
    """확장된 분자 특성 계산 (MACCS keys 포함)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 기본 descriptor
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
    
    # 확장 descriptor (Advanced features)
    if CFG['ADVANCED_FEATURES']:
        try:
            # MACCS keys (166 bits)
            maccs = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            maccs_bits = [int(x) for x in maccs.ToBitString()]
            for i, bit in enumerate(maccs_bits):
                descriptors[f'MACCS_{i}'] = bit
                
            # 추가 2D descriptor
            descriptors.update({
                'VSA_EState1': Descriptors.VSA_EState1(mol),
                'VSA_EState2': Descriptors.VSA_EState2(mol),
                'VSA_EState3': Descriptors.VSA_EState3(mol),
                'Chi0v': Descriptors.Chi0v(mol),
                'Chi1v': Descriptors.Chi1v(mol),
                'Chi2v': Descriptors.Chi2v(mol),
                'Kappa1': Descriptors.Kappa1(mol),
                'Kappa2': Descriptors.Kappa2(mol),
                'Kappa3': Descriptors.Kappa3(mol),
            })
        except:
            # MACCS keys 계산 실패 시 기본값
            for i in range(166):
                descriptors[f'MACCS_{i}'] = 0
    
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

def calculate_sample_weights(molecular_features):
    """Test 분포와 유사한 샘플에 가중치 부여"""
    weights = np.ones(len(molecular_features))
    
    if CFG['SAMPLE_WEIGHTING']:
        # Test 특성: MW 더 높음 (461.69), LogP 더 낮음 (3.59)
        mw = molecular_features['MW'].values
        logp = molecular_features['LogP'].values
        
        # Test-like 조건: MW >= 450 & LogP <= 3.8
        test_like_mask = (mw >= 450) & (logp <= 3.8)
        weights[test_like_mask] = 2.0  # 2배 가중치
        
        # 추가 가중치: MW가 매우 높은 경우 (460+)
        very_high_mw = mw >= 460
        weights[very_high_mw] = weights[very_high_mw] * 1.5
        
        print(f"Test-like 샘플: {test_like_mask.sum()}/{len(weights)} ({test_like_mask.mean()*100:.1f}%)")
        print(f"가중치 범위: {weights.min():.1f} ~ {weights.max():.1f}")
    
    return weights

def create_domain_adaptive_folds(y, molecular_features, groups, n_splits=8):
    """Domain adaptation을 고려한 fold 생성"""
    unique_groups = len(np.unique(groups))
    
    if CFG['GROUP_CV'] and unique_groups >= n_splits:
        # Group K-Fold: 데이터소스별 분리 (그룹 수가 충분할 때만)
        print(f"Group CV 사용: {unique_groups}개 그룹")
        gkf = GroupKFold(n_splits=n_splits)
        return gkf.split(molecular_features, y, groups)
    else:
        # Stratified 방식으로 fallback
        if CFG['GROUP_CV']:
            print(f"그룹 수({unique_groups})가 부족하여 Stratified CV로 변경")
        else:
            print("Stratified CV 사용")
            
        # 층화를 위한 복합 변수 생성
        try:
            pIC50_bins = pd.cut(y, bins=[0, 5, 7, 9, 15], labels=['weak', 'moderate', 'strong', 'very_strong'])
            mw_bins = pd.cut(molecular_features['MW'], bins=4, labels=['small', 'medium', 'large', 'very_large'])
            
            # 데이터 소스도 포함한 복합 층화
            source_labels = ['ChEMBL' if g == 0 else 'PubChem' for g in groups]
            combined_strata = (pIC50_bins.astype(str) + '_' + 
                             mw_bins.astype(str) + '_' + 
                             pd.Series(source_labels))
            
            # 빈도가 너무 낮은 층은 병합
            strata_counts = combined_strata.value_counts()
            min_samples = max(2, len(y) // (n_splits * 10))  # 최소 샘플 수
            
            for strata in strata_counts[strata_counts < min_samples].index:
                # 빈도 낮은 층을 가장 유사한 층으로 병합
                base_strata = '_'.join(strata.split('_')[:-1])  # 소스 제외
                combined_strata = combined_strata.replace(strata, base_strata + '_merged')
            
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=CFG['SEED'])
            return skf.split(molecular_features, combined_strata)
            
        except Exception as e:
            print(f"층화 생성 중 오류, 단순 KFold 사용: {e}")
            # 최후의 수단: 단순 KFold
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=CFG['SEED'])
            return kf.split(molecular_features)

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

# 개선된 Optuna 최적화 함수들
def optimize_extratrees(X, y, cv_folds, sample_weights, save_path):
    if os.path.exists(save_path):
        print(f"기존 최적화 결과 로드: {save_path}")
        return load_optuna_params(save_path)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 800),
            'max_depth': trial.suggest_int('max_depth', 10, 40),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.7, 0.8, 0.9]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': CFG['SEED']
        }
        
        scores = []
        for train_idx, val_idx in cv_folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            w_train = sample_weights[train_idx] if sample_weights is not None else None
            
            model = ExtraTreesRegressor(**params)
            model.fit(X_train, y_train, sample_weight=w_train)
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
            'n_estimators': 400,
            'max_depth': 25,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': CFG['SEED']
        }
        save_optuna_params(default_params, save_path)
        return default_params

def optimize_xgboost_v2(X, y, cv_folds, sample_weights, save_path):
    """XGBoost 완전 재설계 - 정규화 강화"""
    if os.path.exists(save_path):
        print(f"기존 최적화 결과 로드: {save_path}")
        return load_optuna_params(save_path)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),  # 더 많은 트리
            'max_depth': trial.suggest_int('max_depth', 3, 8),  # 더 얕은 트리
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),  # 더 낮은 학습률
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 20),  # 더 강한 정규화
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'random_state': CFG['SEED'],
            'verbosity': 0
        }
        
        scores = []
        for train_idx, val_idx in cv_folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            w_train = sample_weights[train_idx] if sample_weights is not None else None
            
            try:
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, sample_weight=w_train, 
                         eval_set=[(X_val, y_val)], verbose=False)
                y_pred = model.predict(X_val)
            except Exception as e:
                # 파라미터 문제 시 재시도
                params_safe = {k: v for k, v in params.items() if k not in ['early_stopping_rounds']}
                model = xgb.XGBRegressor(**params_safe)
                model.fit(X_train, y_train, sample_weight=w_train)
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
            'n_estimators': 1000,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 5,
            'reg_lambda': 5,
            'min_child_weight': 3,
            'gamma': 1,
            'random_state': CFG['SEED'],
            'verbosity': 0
        }
        save_optuna_params(default_params, save_path)
        return default_params

def optimize_lightgbm_v2(X, y, cv_folds, sample_weights, save_path):
    if os.path.exists(save_path):
        print(f"기존 최적화 결과 로드: {save_path}")
        return load_optuna_params(save_path)
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 15),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'random_state': CFG['SEED'],
            'verbosity': -1
        }
        
        scores = []
        for train_idx, val_idx in cv_folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            w_train = sample_weights[train_idx] if sample_weights is not None else None
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
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
            'n_estimators': 1000,
            'max_depth': 8,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 3,
            'reg_lambda': 3,
            'min_child_samples': 20,
            'random_state': CFG['SEED'],
            'verbosity': -1
        }
        save_optuna_params(default_params, save_path)
        return default_params

def optimize_catboost_v2(X, y, cv_folds, sample_weights, save_path):
    """CatBoost 안전한 구현"""
    if os.path.exists(save_path):
        print(f"기존 최적화 결과 로드: {save_path}")
        return load_optuna_params(save_path)
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 0.95),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 20),
            'random_state': CFG['SEED'],
            'verbose': False,
            'allow_writing_files': False  # 임시 파일 생성 방지
        }
        
        scores = []
        for train_idx, val_idx in cv_folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            w_train = sample_weights[train_idx] if sample_weights is not None else None
            
            try:
                model = cb.CatBoostRegressor(**params)
                model.fit(X_train, y_train, sample_weight=w_train, 
                         eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
                y_pred = model.predict(X_val)
            except Exception as e:
                # early_stopping_rounds 문제 시 안전한 방식
                params_safe = {k: v for k, v in params.items()}
                params_safe['early_stopping_rounds'] = 50
                model = cb.CatBoostRegressor(**params_safe)
                model.fit(X_train, y_train, sample_weight=w_train, eval_set=[(X_val, y_val)], verbose=False)
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
            'iterations': 1000,
            'depth': 6,
            'learning_rate': 0.08,
            'subsample': 0.8,
            'colsample_bylevel': 0.8,
            'l2_leaf_reg': 5,
            'random_state': CFG['SEED'],
            'verbose': False,
            'allow_writing_files': False
        }
        save_optuna_params(default_params, save_path)
        return default_params

try:
    print("=" * 60)
    print("Try #3: Domain Adaptation + 기술적 완성도")
    print("=" * 60)

    # 1. 데이터 로드 및 전처리
    print("\n1. 데이터 로드 및 전처리")
    print("-" * 50)

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

    # 2. 확장된 분자 특성 계산
    print("\n2. 확장된 분자 특성 계산")
    print("-" * 50)

    molecular_descriptors = []
    valid_indices = []

    for idx, smiles in enumerate(total['smiles']):
        descriptors = calculate_molecular_descriptors_extended(smiles)
        if descriptors is not None:
            molecular_descriptors.append(descriptors)
            valid_indices.append(idx)

    total = total.iloc[valid_indices].reset_index(drop=True)
    molecular_df = pd.DataFrame(molecular_descriptors)

    print(f"계산된 특성 수: {len(molecular_df.columns)}")
    if CFG['ADVANCED_FEATURES']:
        print(f"MACCS keys 포함: 166개")

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

    # 분자 특성과 fingerprint 결합
    if CFG['USE_MOLECULAR_FEATURES']:
        X_combined = np.hstack([X_fingerprint, molecular_df.values])
        feature_names = [f'morgan_{i}' for i in range(CFG['NBITS'])] + list(molecular_df.columns)
    else:
        X_combined = X_fingerprint
        feature_names = [f'morgan_{i}' for i in range(CFG['NBITS'])]

    print(f"결합된 특성 수: {X_combined.shape[1]}")

    # 4. 특성 스케일링
    print("\n4. 특성 스케일링")
    print("-" * 50)

    if CFG['SCALE_FEATURES']:
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_combined)
        save_model(scaler, 'output/try3/scaler.pkl')
    else:
        X_scaled = X_combined

    # 5. 특성 선택
    print("\n5. 특성 선택")
    print("-" * 50)

    y = total['pIC50'].values
    
    if CFG['FEATURE_SELECTION']:
        selector = SelectKBest(score_func=mutual_info_regression, k=CFG['N_FEATURES'])
        X_selected = selector.fit_transform(X_scaled, y)
        selected_features = np.array(feature_names)[selector.get_support()]
        save_model(selector, 'output/try3/feature_selector.pkl')
        print(f"선택된 특성: {X_selected.shape[1]} / {X_scaled.shape[1]}")
    else:
        X_selected = X_scaled
        selected_features = feature_names

    # 6. Sample Weighting
    print("\n6. Sample Weighting")
    print("-" * 50)

    sample_weights = calculate_sample_weights(molecular_df)

    # 7. Cross Validation 설정
    print("\n7. Cross Validation 설정")
    print("-" * 50)

    # Group 정보 (데이터 소스)
    groups = total['source'].map({'ChEMBL': 0, 'PubChem': 1}).values
    
    cv_folds = list(create_domain_adaptive_folds(y, molecular_df, groups, CFG['N_FOLDS']))
    print(f"CV Folds: {len(cv_folds)}")

    # 8. 모델 최적화
    print("\n8. 모델 하이퍼파라미터 최적화")
    print("-" * 50)

    models_params = {}

    # ExtraTrees
    print("\n8.1 ExtraTrees 최적화")
    models_params['ExtraTrees'] = optimize_extratrees(
        X_selected, y, cv_folds, sample_weights, 'output/try3/extratrees_params.json'
    )

    # XGBoost
    print("\n8.2 XGBoost 최적화")
    models_params['XGBoost'] = optimize_xgboost_v2(
        X_selected, y, cv_folds, sample_weights, 'output/try3/xgboost_params.json'
    )

    # LightGBM
    print("\n8.3 LightGBM 최적화")
    models_params['LightGBM'] = optimize_lightgbm_v2(
        X_selected, y, cv_folds, sample_weights, 'output/try3/lightgbm_params.json'
    )

    # CatBoost
    print("\n8.4 CatBoost 최적화")
    models_params['CatBoost'] = optimize_catboost_v2(
        X_selected, y, cv_folds, sample_weights, 'output/try3/catboost_params.json'
    )

    # 최적화 결과 저장
    save_config(models_params, 'output/try3/all_optimized_params.json')

    # 9. 모델 학습 및 평가
    print("\n9. 모델 학습 및 평가")
    print("-" * 50)

    models = {}
    model_scores = {}
    model_predictions = {}
    fold_predictions = {}  # fold별 예측 저장

    for model_name, params in models_params.items():
        print(f"\n9.{list(models_params.keys()).index(model_name)+1} {model_name} 학습 및 평가")
        
        cv_scores = []
        cv_preds = np.zeros(len(y))
        fold_preds = []  # 각 fold의 예측 저장
        
        for fold, (train_idx, val_idx) in enumerate(cv_folds):
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            w_train = sample_weights[train_idx] if sample_weights is not None else None
            
            # 모델 생성 및 학습
            if model_name == 'ExtraTrees':
                model = ExtraTreesRegressor(**params)
                model.fit(X_train, y_train, sample_weight=w_train)
            elif model_name == 'XGBoost':
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
            elif model_name == 'LightGBM':
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, sample_weight=w_train, 
                         eval_set=[(X_val, y_val)], 
                         callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
            elif model_name == 'CatBoost':
                model = cb.CatBoostRegressor(**params)
                model.fit(X_train, y_train, sample_weight=w_train, 
                         eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
            
            # 예측
            y_pred = model.predict(X_val)
            cv_preds[val_idx] = y_pred
            fold_preds.append((val_idx, y_pred))  # fold별 예측 저장
            
            # 평가
            y_val_ic50 = pIC50_to_IC50(y_val)
            y_pred_ic50 = pIC50_to_IC50(y_pred)
            
            norm_rmse, corr_sq, final_score = calculate_metrics(y_val_ic50, y_pred_ic50, y_val, y_pred)
            cv_scores.append(final_score)
            
            print(f"  Fold {fold+1}: {final_score:.4f}")
            
            # 모델 저장 (첫 번째 fold만)
            if fold == 0:
                save_model(model, f'output/try3/{model_name.lower()}_model.pkl')
        
        # 전체 CV 성능
        y_ic50 = pIC50_to_IC50(y)
        cv_preds_ic50 = pIC50_to_IC50(cv_preds)
        norm_rmse, corr_sq, final_score = calculate_metrics(y_ic50, cv_preds_ic50, y, cv_preds)
        
        model_scores[model_name] = {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'oof_score': final_score
        }
        model_predictions[model_name] = cv_preds
        fold_predictions[model_name] = fold_preds  # fold별 예측 저장
        
        print(f"  CV Mean: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        print(f"  OOF Score: {final_score:.4f}")

    # 10. 앙상블
    print("\n10. 앙상블")
    print("-" * 50)

    # 각 fold별로 앙상블 성능 계산
    ensemble_cv_scores = []
    ensemble_oof_preds = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(cv_folds):
        # 해당 fold의 각 모델 예측 수집
        fold_model_preds = []
        for model_name in models_params.keys():
            # 해당 fold의 예측 찾기
            for fold_idx, fold_pred in fold_predictions[model_name]:
                if np.array_equal(fold_idx, val_idx):
                    fold_model_preds.append(fold_pred)
                    break
        
        # 앙상블 예측 (단순 평균)
        ensemble_fold_pred = np.mean(fold_model_preds, axis=0)
        ensemble_oof_preds[val_idx] = ensemble_fold_pred
        
        # 해당 fold 성능 계산
        y_val = y[val_idx]
        y_val_ic50 = pIC50_to_IC50(y_val)
        ensemble_fold_pred_ic50 = pIC50_to_IC50(ensemble_fold_pred)
        
        norm_rmse, corr_sq, fold_score = calculate_metrics(y_val_ic50, ensemble_fold_pred_ic50, y_val, ensemble_fold_pred)
        ensemble_cv_scores.append(fold_score)
        
        print(f"  Ensemble Fold {fold+1}: {fold_score:.4f}")
    
    # 전체 OOF 성능
    y_ic50 = pIC50_to_IC50(y)
    ensemble_oof_ic50 = pIC50_to_IC50(ensemble_oof_preds)
    norm_rmse, corr_sq, ensemble_oof_score = calculate_metrics(y_ic50, ensemble_oof_ic50, y, ensemble_oof_preds)
    
    model_scores['Ensemble'] = {
        'cv_mean': np.mean(ensemble_cv_scores),
        'cv_std': np.std(ensemble_cv_scores),
        'oof_score': ensemble_oof_score
    }

    print(f"  Ensemble CV Mean: {np.mean(ensemble_cv_scores):.4f} ± {np.std(ensemble_cv_scores):.4f}")
    print(f"  Ensemble OOF Score: {ensemble_oof_score:.4f}")

    # 11. 최종 예측
    print("\n11. 최종 예측")
    print("-" * 50)

    # Test 데이터 로드
    test_df = pd.read_csv('data/test.csv')
    
    # Test 데이터 전처리
    test_descriptors = []
    test_valid_indices = []
    
    for idx, smiles in enumerate(test_df['smiles']):
        descriptors = calculate_molecular_descriptors_extended(smiles)
        if descriptors is not None:
            test_descriptors.append(descriptors)
            test_valid_indices.append(idx)
    
    test_df = test_df.iloc[test_valid_indices].reset_index(drop=True)
    test_molecular_df = pd.DataFrame(test_descriptors)
    
    # Test fingerprint 계산
    test_fingerprints = []
    for smiles in test_df['smiles']:
        fp = smiles_to_fingerprint(smiles, radius=CFG['MORGAN_RADIUS'], nbits=CFG['NBITS'])
        test_fingerprints.append(fp)
    
    X_test_fingerprint = np.array(test_fingerprints)
    
    # Test 특성 결합
    if CFG['USE_MOLECULAR_FEATURES']:
        X_test_combined = np.hstack([X_test_fingerprint, test_molecular_df.values])
    else:
        X_test_combined = X_test_fingerprint
    
    # Test 스케일링
    if CFG['SCALE_FEATURES']:
        X_test_scaled = scaler.transform(X_test_combined)
    else:
        X_test_scaled = X_test_combined
    
    # Test 특성 선택
    if CFG['FEATURE_SELECTION']:
        X_test_selected = selector.transform(X_test_scaled)
    else:
        X_test_selected = X_test_scaled
    
    # 각 모델로 예측
    test_predictions = {}
    for model_name in models_params.keys():
        model = load_model(f'output/try3/{model_name.lower()}_model.pkl')
        test_pred = model.predict(X_test_selected)
        test_predictions[model_name] = test_pred
    
    # 앙상블 예측도 추가
    ensemble_test_pred = np.mean([pred for pred in test_predictions.values()], axis=0)
    test_predictions['Ensemble'] = ensemble_test_pred
    
    # Best CV model 선택 (CV_Score 기준)
    best_model_name = max(model_scores.items(), key=lambda x: x[1]['cv_mean'])[0]
    print(f"Best CV model: {best_model_name} (CV_Score: {model_scores[best_model_name]['cv_mean']:.4f})")
    
    # Best CV model 예측 사용
    if best_model_name == 'Ensemble':
        test_ic50_pred = pIC50_to_IC50(ensemble_test_pred)
    else:
        test_ic50_pred = pIC50_to_IC50(test_predictions[best_model_name])
    
    print(f"선택된 모델: {best_model_name}")

    # 12. 결과 저장
    print("\n12. 결과 저장")
    print("-" * 50)

    # 성능 요약
    results_df = pd.DataFrame([
        {
            'Model': name,
            'CV_Score': scores['cv_mean'],
            'CV_Std': scores['cv_std'],
            'OOF_Score': scores['oof_score']
        }
        for name, scores in model_scores.items()
    ])
    
    results_df = results_df.sort_values('CV_Score', ascending=False)  # CV_Score 기준으로 정렬
    print("\n성능 요약 (CV_Score 기준):")
    print(results_df.to_string(index=False))
    
    results_df.to_csv('output/try3/model_performance.csv', index=False)
    
    # Submission 파일
    submission = test_df[['ID']].copy()
    submission['ASK1_IC50_nM'] = test_ic50_pred
    submission.to_csv('output/try3/submission.csv', index=False)
    
    # 설정 저장
    save_config(CFG, 'output/try3/config.json')
    
    print(f"\n제출 파일 생성 완료: output/try3/submission.csv")
    print(f"사용된 모델: {best_model_name}")
    print(f"예측값 범위: {test_ic50_pred.min():.2f} ~ {test_ic50_pred.max():.2f} nM")

except Exception as e:
    print(f"오류 발생: {e}")
    import traceback
    traceback.print_exc()

finally:
    # 로그 파일 정리
    sys.stdout = sys.__stdout__
    logger.close()
    print(f"로그 저장: {log_filename}") 