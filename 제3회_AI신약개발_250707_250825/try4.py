import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import catboost as cb
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# 출력 디렉토리 생성
os.makedirs('output/try4', exist_ok=True)

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
log_filename = f'output/try4/print_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logger = Logger(log_filename)
sys.stdout = logger

CFG = {
    'NBITS': 1024,                      # 차원 축소 (2048 -> 1024)
    'SEED': 42,
    'N_FOLDS': 2,                       # Leave-One-Source-Out CV
    'OPTUNA_TRIALS': 100,
    'MORGAN_RADIUS': 2,
    'REMOVE_OUTLIERS': True,            # pIC50 기반 이상치 제거는 유지
    'USE_MOLECULAR_FEATURES': False,    # 핵심 변경: Morgan Fingerprint만 사용
    'SCALE_FEATURES': False,            # Fingerprint는 0/1이므로 스케일링 불필요
    'FEATURE_SELECTION': False,         # 특성 선택 미사용
    'GROUP_CV': True,                   # 데이터소스 기반 Group CV (Leave-One-Source-Out)
    'EARLY_STOPPING_PATIENCE': 20,      # 더 엄격한 Early stopping
}

def seed_everything(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED'])

def save_config(config, filepath):
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)

def save_optuna_params(params, filepath):
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)

def load_optuna_params(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

def load_model(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

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

def create_cv_folds(y, groups, n_splits, use_group_cv, seed):
    """CV fold 생성: Group CV 또는 Stratified CV"""
    unique_groups = len(np.unique(groups))
    
    if use_group_cv and unique_groups >= n_splits:
        print(f"Group CV 사용 (Leave-One-Source-Out): {unique_groups}개 그룹, {n_splits}-fold")
        gkf = GroupKFold(n_splits=n_splits)
        return gkf.split(np.zeros(len(y)), y, groups)
    else:
        if use_group_cv:
            print(f"그룹 수({unique_groups})가 부족하여 Stratified CV로 변경")
        print(f"Stratified CV 사용, {n_splits}-fold")
        
        pIC50_bins = pd.cut(y, bins=5, labels=False, include_lowest=True)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return skf.split(np.zeros(len(y)), pIC50_bins)

def calculate_metrics(y_true_ic50, y_pred_ic50, y_true_pic50, y_pred_pic50):
    rmse = np.sqrt(mean_squared_error(y_true_ic50, y_pred_ic50))
    y_range = np.max(y_true_ic50) - np.min(y_true_ic50)
    normalized_rmse = rmse / y_range if y_range > 0 else 0
    
    corr, _ = pearsonr(y_true_pic50, y_pred_pic50)
    correlation_squared = corr ** 2
    
    A = normalized_rmse
    B = correlation_squared
    final_score = 0.4 * (1 - min(A, 1)) + 0.6 * B
    
    return normalized_rmse, correlation_squared, final_score

def optimize_catboost(X, y, cv_folds, save_path):
    """CatBoost 최적화 - 강력한 정규화"""
    if os.path.exists(save_path):
        print(f"기존 최적화 결과 로드: {save_path}")
        return load_optuna_params(save_path)
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2500),
            'depth': trial.suggest_int('depth', 4, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 0.9),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 5, 50), # L2 정규화 강화
            'random_state': CFG['SEED'],
            'verbose': False,
            'allow_writing_files': False
        }
        
        scores = []
        for train_idx, val_idx in cv_folds:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = cb.CatBoostRegressor(**params)
            model.fit(X_train, y_train, 
                      eval_set=[(X_val, y_val)], 
                      early_stopping_rounds=CFG['EARLY_STOPPING_PATIENCE'], 
                      verbose=False)
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
        default_params = { 'iterations': 1000, 'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 10 }
        save_optuna_params(default_params, save_path)
        return default_params

try:
    print("=" * 60)
    print("Try #4: Generalization 중심 접근 (단순화 + 정규화)")
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
    total = total.drop_duplicates(subset='smiles').reset_index(drop=True)

    print(f"전처리 후 데이터: {len(total):,} 개")
    
    # pIC50 변환 및 이상치 제거
    total['pIC50'] = IC50_to_pIC50(total['ic50_nM'])
    
    if CFG['REMOVE_OUTLIERS']:
        outliers_pic50 = detect_outliers_iqr(total, 'pIC50', factor=2.0)
        total = total[~outliers_pic50].reset_index(drop=True)
        print(f"이상치 제거 후 데이터: {len(total):,} 개")

    # 2. Feature Engineering (Morgan Fingerprint only)
    print("\n2. Feature Engineering (Morgan Fingerprint only)")
    print("-" * 50)

    fingerprints = []
    for smiles in total['smiles']:
        fp = smiles_to_fingerprint(smiles, radius=CFG['MORGAN_RADIUS'], nbits=CFG['NBITS'])
        fingerprints.append(fp)

    X = np.array(fingerprints)
    y = total['pIC50'].values

    print(f"최종 특성 수: {X.shape[1]}")

    # 3. Cross Validation 설정
    print("\n3. Cross Validation 설정")
    print("-" * 50)

    groups = total['source'].map({'ChEMBL': 0, 'PubChem': 1}).values
    cv_folds = list(create_cv_folds(y, groups, CFG['N_FOLDS'], CFG['GROUP_CV'], CFG['SEED']))
    print(f"CV Folds: {len(cv_folds)}")

    # 4. 모델 최적화
    print("\n4. CatBoost 하이퍼파라미터 최적화")
    print("-" * 50)

    catboost_params = optimize_catboost(X, y, cv_folds, 'output/try4/catboost_params.json')

    # 5. 모델 학습 및 평가
    print("\n5. CatBoost 모델 학습 및 평가")
    print("-" * 50)
    
    cv_scores = []
    oof_preds = np.zeros(len(y))
    
    for fold, (train_idx, val_idx) in enumerate(cv_folds):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = cb.CatBoostRegressor(**catboost_params)
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], 
                  early_stopping_rounds=CFG['EARLY_STOPPING_PATIENCE'], 
                  verbose=False)
        
        # 예측
        y_pred = model.predict(X_val)
        oof_preds[val_idx] = y_pred
        
        # 평가
        y_val_ic50 = pIC50_to_IC50(y_val)
        y_pred_ic50 = pIC50_to_IC50(y_pred)
        
        _, _, final_score = calculate_metrics(y_val_ic50, y_pred_ic50, y_val, y_pred)
        cv_scores.append(final_score)
        
        print(f"  Fold {fold+1}: {final_score:.4f}")
    
    # 전체 CV 성능
    y_ic50 = pIC50_to_IC50(y)
    oof_preds_ic50 = pIC50_to_IC50(oof_preds)
    norm_rmse, corr_sq, final_oof_score = calculate_metrics(y_ic50, oof_preds_ic50, y, oof_preds)
    
    print(f"  CV Mean Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"  OOF Score: {final_oof_score:.4f}")

    # 6. 전체 데이터로 최종 모델 학습
    print("\n6. 전체 데이터로 최종 모델 학습")
    print("-" * 50)
    
    final_model = cb.CatBoostRegressor(**catboost_params)
    final_model.fit(X, y, verbose=False)
    save_model(final_model, 'output/try4/catboost_final_model.pkl')
    print("최종 모델 학습 및 저장 완료.")

    # 7. 최종 예측
    print("\n7. 최종 예측")
    print("-" * 50)

    test_df = pd.read_csv('data/test.csv')
    
    test_fingerprints = []
    for smiles in test_df['smiles']:
        fp = smiles_to_fingerprint(smiles, radius=CFG['MORGAN_RADIUS'], nbits=CFG['NBITS'])
        test_fingerprints.append(fp)
    
    X_test = np.array(test_fingerprints)
    
    # 최종 모델로 예측
    test_pic50_pred = final_model.predict(X_test)
    test_ic50_pred = pIC50_to_IC50(test_pic50_pred)
    
    # 8. 결과 저장
    print("\n8. 결과 저장")
    print("-" * 50)

    # 성능 요약
    results_df = pd.DataFrame([{
        'Model': 'CatBoost',
        'CV_Score': np.mean(cv_scores),
        'CV_Std': np.std(cv_scores),
        'OOF_Score': final_oof_score
    }])
    print("\n성능 요약:")
    print(results_df.to_string(index=False))
    results_df.to_csv('output/try4/model_performance.csv', index=False)
    
    # Submission 파일
    submission = test_df[['ID']].copy()
    submission['ASK1_IC50_nM'] = test_ic50_pred
    submission.to_csv('output/try4/submission.csv', index=False)
    
    save_config(CFG, 'output/try4/config.json')
    
    print(f"\n제출 파일 생성 완료: output/try4/submission.csv")
    print(f"예측값 범위: {test_ic50_pred.min():.2f} ~ {test_ic50_pred.max():.2f} nM")

except Exception as e:
    print(f"오류 발생: {e}")
    import traceback
    traceback.print_exc()

finally:
    sys.stdout = sys.__stdout__
    logger.close()
    print(f"로그 저장: {log_filename}") 