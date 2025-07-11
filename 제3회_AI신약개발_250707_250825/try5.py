import pandas as pd
import numpy as np
import os
import sys
import json
import pickle
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, Crippen, rdMolDescriptors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# 출력 디렉토리 생성
os.makedirs('output/try5', exist_ok=True)

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
log_filename = f'output/try5/print_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logger = Logger(log_filename)
sys.stdout = logger

CFG = {
    'NBITS': 2048,
    'SEED': 42,
    'N_FOLDS': 8,                       # 안정적인 평가를 위해 8-Fold 복귀
    'OPTUNA_TRIALS': 100,
    'MORGAN_RADIUS': 2,
    'REMOVE_OUTLIERS': True,
    'USE_MOLECULAR_FEATURES': True,
    'ADVERSARIAL_VALIDATION': True,     # 핵심 전략: Adversarial Validation
    'ADV_VAL_DROP_N': 5,               # 제거할 불안정 피처 수 (20 -> 5로 수정)
    'EARLY_STOPPING_PATIENCE': 30,
}

def seed_everything(seed):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED'])

# --- 유틸리티 함수 (저장/로드 등) ---
def save_config(config, filepath):
    with open(filepath, 'w') as f: json.dump(config, f, indent=2)

def save_optuna_params(params, filepath):
    with open(filepath, 'w') as f: json.dump(params, f, indent=2)

def load_optuna_params(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as f: return json.load(f)
    return None

def save_model(model, filepath):
    with open(filepath, 'wb') as f: pickle.dump(model, f)

def load_model(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f: return pickle.load(f)
    return None

# --- 피처 엔지니어링 함수 ---
def calculate_molecular_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    
    return {
        'MW': Descriptors.MolWt(mol), 'LogP': Crippen.MolLogP(mol), 'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol), 'TPSA': Descriptors.TPSA(mol), 'Rotatable': Descriptors.NumRotatableBonds(mol),
        'Aromatic_Rings': Descriptors.NumAromaticRings(mol), 'Heavy_Atoms': Descriptors.HeavyAtomCount(mol),
        'FractionCsp3': Descriptors.FractionCSP3(mol), 'RingCount': Descriptors.RingCount(mol),
    }

def process_smiles_to_features(df, is_train=True):
    # 분자 특성 계산
    descriptors_list = []
    valid_indices = []
    for idx, smiles in enumerate(df['smiles']):
        descriptors = calculate_molecular_descriptors(smiles)
        if descriptors is not None:
            descriptors_list.append(descriptors)
            valid_indices.append(idx)
    
    df = df.iloc[valid_indices].reset_index(drop=True)
    molecular_df = pd.DataFrame(descriptors_list)

    # pIC50 변환 및 이상치 제거 (Train 데이터에만 적용)
    if is_train:
        df['pIC50'] = IC50_to_pIC50(df['ic50_nM'])
        if CFG['REMOVE_OUTLIERS']:
            outliers = detect_outliers_iqr(df, 'pIC50', factor=2.0)
            df = df[~outliers].reset_index(drop=True)
            molecular_df = molecular_df[~outliers].reset_index(drop=True)
    
    # Morgan Fingerprint 계산
    fingerprints = np.array([
        smiles_to_fingerprint(s, CFG['MORGAN_RADIUS'], CFG['NBITS']) for s in df['smiles']
    ])
    
    return df, molecular_df, fingerprints

def smiles_to_fingerprint(smiles, radius, nbits):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits).GetFingerprintAsNumPy(mol))
    return np.zeros(nbits)

# --- 데이터 변환 및 평가 함수 ---
def IC50_to_pIC50(ic50): return 9 - np.log10(np.clip(ic50, 1e-3, None))
def pIC50_to_IC50(pic50): return 10**(9 - pic50)
def detect_outliers_iqr(data, column, factor=1.5):
    q1, q3 = data[column].quantile([0.25, 0.75])
    return (data[column] < q1 - factor * (q3 - q1)) | (data[column] > q3 + factor * (q3 - q1))

def calculate_metrics(y_true_ic50, y_pred_ic50, y_true_pic50, y_pred_pic50):
    rmse = np.sqrt(mean_squared_error(y_true_ic50, y_pred_ic50))
    y_range = np.max(y_true_ic50) - np.min(y_true_ic50)
    norm_rmse = rmse / y_range if y_range > 0 else 0
    corr_sq = pearsonr(y_true_pic50, y_pred_pic50)[0]**2
    final_score = 0.4 * (1 - min(norm_rmse, 1)) + 0.6 * corr_sq
    return final_score

# --- 핵심 로직: Adversarial Validation ---
def run_adversarial_validation(train_feats, test_feats, n_drop):
    print(f"\nAdversarial Validation 수행 (상위 {n_drop}개 피처 제거)")
    
    train_feats['is_test'] = 0
    test_feats['is_test'] = 1
    
    combined = pd.concat([train_feats, test_feats], ignore_index=True)
    y = combined['is_test']
    X = combined.drop('is_test', axis=1)
    
    model = lgb.LGBMClassifier(random_state=CFG['SEED'], n_estimators=200)
    model.fit(X, y)
    
    importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
    importances = importances.sort_values('importance', ascending=False)
    
    features_to_drop = importances.head(n_drop)['feature'].tolist()
    
    print("제거될 피처 (Train/Test 분포 차이가 큰 피처):")
    print(features_to_drop)
    
    return features_to_drop

# --- 모델 최적화 함수 ---
def optimize_catboost(X, y, cv_folds, save_path):
    if os.path.exists(save_path):
        return load_optuna_params(save_path)
    
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2000),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 30),
            'random_state': CFG['SEED'], 'verbose': False, 'allow_writing_files': False
        }
        scores = [
            np.sqrt(mean_squared_error(
                y[val_idx],
                cb.CatBoostRegressor(**params).fit(
                    X[train_idx], y[train_idx],
                    eval_set=[(X[val_idx], y[val_idx])],
                    early_stopping_rounds=CFG['EARLY_STOPPING_PATIENCE'],
                    verbose=False
                ).predict(X[val_idx])
            )) for train_idx, val_idx in cv_folds
        ]
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=CFG['SEED']))
    study.optimize(objective, n_trials=CFG['OPTUNA_TRIALS'])
    save_optuna_params(study.best_params, save_path)
    return study.best_params

try:
    print("=" * 60)
    print("Try #5: Adversarial Validation 기반 Feature Engineering")
    print("=" * 60)

    # 1. 데이터 로드
    chembl = pd.read_csv("data/ChEMBL_ASK1(IC50).csv", sep=';')
    chembl.columns = chembl.columns.str.strip().str.replace('"', '')
    chembl = chembl[chembl['Standard Type'] == 'IC50'][['Smiles', 'Standard Value']].rename(columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'}).dropna()
    chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')
    
    pubchem = pd.read_csv("data/Pubchem_ASK1.csv", low_memory=False)
    pubchem = pubchem[['SMILES', 'Activity_Value']].rename(columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'}).dropna()
    pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')
    
    train_df = pd.concat([chembl, pubchem]).dropna(subset=['ic50_nM']).drop_duplicates(subset='smiles').reset_index(drop=True)
    test_df = pd.read_csv('data/test.csv')

    # 2. 피처 생성 (Train/Test)
    print("\n1. 피처 생성")
    train_df, train_molecular_df, X_train_fp = process_smiles_to_features(train_df, is_train=True)
    test_df, test_molecular_df, X_test_fp = process_smiles_to_features(test_df, is_train=False)
    print(f"학습 데이터: {len(train_df)}, 테스트 데이터: {len(test_df)}")

    # 3. Adversarial Validation
    if CFG['ADVERSARIAL_VALIDATION']:
        features_to_drop = run_adversarial_validation(
            train_molecular_df.copy(), test_molecular_df.copy(), CFG['ADV_VAL_DROP_N']
        )
        train_molecular_df.drop(columns=features_to_drop, inplace=True)
        test_molecular_df.drop(columns=features_to_drop, inplace=True)
        print(f"분자 특성 차원: {train_molecular_df.shape[1]}")

    # 4. 최종 피처셋 결합
    X_train = np.hstack([X_train_fp, train_molecular_df.values])
    X_test = np.hstack([X_test_fp, test_molecular_df.values])
    y_train = train_df['pIC50'].values
    print(f"최종 피처 차원: {X_train.shape[1]}")

    # 5. 모델 최적화
    print("\n2. 모델 최적화")
    skf = StratifiedKFold(n_splits=CFG['N_FOLDS'], shuffle=True, random_state=CFG['SEED'])
    cv_folds = list(skf.split(X_train, pd.cut(y_train, 5, labels=False)))
    
    params = optimize_catboost(X_train, y_train, cv_folds, 'output/try5/catboost_params.json')
    
    # 6. 모델 학습 및 CV 평가
    print("\n3. 모델 학습 및 CV 평가")
    cv_scores = []
    oof_preds = np.zeros(len(y_train))

    for fold, (train_idx, val_idx) in enumerate(cv_folds):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        model = cb.CatBoostRegressor(**params)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  early_stopping_rounds=CFG['EARLY_STOPPING_PATIENCE'],
                  verbose=False)
        
        y_pred_fold = model.predict(X_val_fold)
        oof_preds[val_idx] = y_pred_fold
        
        y_val_ic50 = pIC50_to_IC50(y_val_fold)
        y_pred_ic50 = pIC50_to_IC50(y_pred_fold)

        fold_score = calculate_metrics(y_val_ic50, y_pred_ic50, y_val_fold, y_pred_fold)
        cv_scores.append(fold_score)
        print(f"  Fold {fold+1}: {fold_score:.4f}")

    y_train_ic50 = pIC50_to_IC50(y_train)
    oof_preds_ic50 = pIC50_to_IC50(oof_preds)
    oof_score = calculate_metrics(y_train_ic50, oof_preds_ic50, y_train, oof_preds)
    
    print(f"\n  CV Mean Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    print(f"  OOF Score: {oof_score:.4f}")

    # 7. 전체 데이터로 최종 모델 학습
    print("\n4. 전체 데이터로 최종 모델 학습")
    final_model = cb.CatBoostRegressor(**params)
    final_model.fit(X_train, y_train, verbose=False)
    save_model(final_model, 'output/try5/catboost_model.pkl')
    print("최종 모델 학습 및 저장 완료.")

    # 8. 예측 및 제출
    print("\n5. 예측 및 제출")
    test_pic50_pred = final_model.predict(X_test)
    test_ic50_pred = pIC50_to_IC50(test_pic50_pred)
    
    submission = test_df[['ID']].copy()
    submission['ASK1_IC50_nM'] = test_ic50_pred
    submission.to_csv('output/try5/submission.csv', index=False)
    
    save_config(CFG, 'output/try5/config.json')

    # 9. 결과 요약 저장
    print("\n6. 결과 요약 저장")
    results_df = pd.DataFrame([{
        'Model': 'CatBoost_AdvVal',
        'CV_Score': np.mean(cv_scores),
        'CV_Std': np.std(cv_scores),
        'OOF_Score': oof_score
    }])
    print("\n성능 요약:")
    print(results_df.to_string(index=False))
    results_df.to_csv('output/try5/model_performance.csv', index=False)
    
    print("\n제출 파일 생성 완료: output/try5/submission.csv")

except Exception as e:
    print(f"오류 발생: {e}")
    import traceback
    traceback.print_exc()

finally:
    sys.stdout = sys.__stdout__
    logger.close()
    print(f"로그 저장: {log_filename}") 