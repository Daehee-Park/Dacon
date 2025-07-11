import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
import os
import json
import pickle
import random
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 설정
CFG = {
    'NBITS': 2048,
    'SEED': 42,
    'N_FOLDS': 5,
    'OPTUNA_TRIALS': 50,
    'OUTPUT_DIR': 'output/try6',
    'EXPERIMENT_NAME': 'try6_back_to_basics'
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def setup_output_directory():
    """출력 디렉토리 구조 생성"""
    dirs = [
        CFG['OUTPUT_DIR'],
        f"{CFG['OUTPUT_DIR']}/model_optimization",
        f"{CFG['OUTPUT_DIR']}/cv_results",
        f"{CFG['OUTPUT_DIR']}/cv_results/fold_predictions",
        f"{CFG['OUTPUT_DIR']}/models",
        f"{CFG['OUTPUT_DIR']}/models/all_models"
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def log_message(message, log_file=None):
    """로그 메시지 출력 및 파일 저장"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')

def smiles_to_fingerprint(smiles):
    """SMILES를 Morgan fingerprint로 변환"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=CFG['NBITS'])
        fp = morgan_gen.GetFingerprintAsNumPy(mol)
        return fp
    else:
        return np.zeros((CFG['NBITS'],))

def calculate_molecular_properties(smiles):
    """핵심 분자 특성 5개 계산"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return {
            'MW': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'HBD': Descriptors.NumHDonors(mol),
            'HBA': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol)
        }
    else:
        return {
            'MW': 0, 'LogP': 0, 'HBD': 0, 'HBA': 0, 'TPSA': 0
        }

def IC50_to_pIC50(ic50_nM):
    ic50_nM = np.clip(ic50_nM, 1e-10, None)
    return 9 - np.log10(ic50_nM)

def pIC50_to_IC50(pIC50):
    return 10 ** (9 - pIC50)

def calculate_normalized_rmse(y_true_ic50, y_pred_ic50):
    """Calculate normalized RMSE on IC50 scale"""
    rmse = np.sqrt(mean_squared_error(y_true_ic50, y_pred_ic50))
    y_range = np.max(y_true_ic50) - np.min(y_true_ic50)
    return rmse / y_range

def calculate_correlation(y_true_pic50, y_pred_pic50):
    """Calculate correlation coefficient squared"""
    corr, _ = pearsonr(y_true_pic50, y_pred_pic50)
    return corr ** 2

def calculate_final_score(normalized_rmse, correlation_squared):
    """Calculate final score according to competition rules"""
    A = normalized_rmse
    B = correlation_squared
    score = 0.4 * (1 - min(A, 1)) + 0.6 * B
    return score

def remove_outliers(df, column='pIC50', factor=3.0):
    """IQR 기반 이상치 제거"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    before_count = len(df)
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    after_count = len(df_clean)
    
    return df_clean, (before_count - after_count)

def load_and_preprocess_data():
    """데이터 로딩 및 전처리"""
    log_file = f"{CFG['OUTPUT_DIR']}/preprocessing_log.txt"
    
    log_message("=== 데이터 로딩 및 전처리 시작 ===", log_file)
    
    # 데이터 로드
    chembl = pd.read_csv("data/ChEMBL_ASK1(IC50).csv", sep=';')
    pubchem = pd.read_csv("data/Pubchem_ASK1.csv", low_memory=False)
    
    # ChEMBL 전처리
    chembl.columns = chembl.columns.str.strip().str.replace('"', '')
    chembl = chembl[chembl['Standard Type'] == 'IC50']
    chembl = chembl[['Smiles', 'Standard Value']].rename(columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'}).dropna()
    chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')
    chembl['pIC50'] = IC50_to_pIC50(chembl['ic50_nM'])
    chembl['source'] = 'ChEMBL'
    
    # PubChem 전처리
    pubchem = pubchem[['SMILES', 'Activity_Value']].rename(columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'}).dropna()
    pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')
    pubchem['pIC50'] = IC50_to_pIC50(pubchem['ic50_nM'])
    pubchem['source'] = 'PubChem'
    
    # 데이터 합치기
    total = pd.concat([chembl, pubchem], ignore_index=True)
    total = total.drop_duplicates(subset='smiles')
    total = total[total['ic50_nM'] > 0].dropna()
    
    log_message(f"초기 데이터 크기: {len(total)}", log_file)
    
    # 이상치 제거
    total_clean, outliers_removed = remove_outliers(total, 'pIC50', factor=3.0)
    log_message(f"이상치 제거: {outliers_removed}개 제거, 남은 데이터: {len(total_clean)}", log_file)
    
    # 분자 특성 계산
    log_message("분자 특성 계산 중...", log_file)
    mol_props = total_clean['smiles'].apply(calculate_molecular_properties)
    mol_props_df = pd.DataFrame(mol_props.tolist())
    
    # Morgan fingerprint 계산
    log_message("Morgan fingerprint 계산 중...", log_file)
    total_clean['Fingerprint'] = total_clean['smiles'].apply(smiles_to_fingerprint)
    total_clean = total_clean[total_clean['Fingerprint'].notnull()]
    
    log_message(f"최종 데이터 크기: {len(total_clean)}", log_file)
    log_message("=== 데이터 전처리 완료 ===", log_file)
    
    return total_clean, mol_props_df

def create_feature_matrix(df, mol_props_df):
    """피처 매트릭스 생성 (Morgan FP + 분자 특성)"""
    # Morgan fingerprint
    X_fp = np.stack(df['Fingerprint'].values)
    
    # 분자 특성 정규화
    scaler = StandardScaler()
    X_props = scaler.fit_transform(mol_props_df)
    
    # 결합
    X = np.hstack([X_fp, X_props])
    
    return X, scaler

def load_existing_optimization_results(model_name):
    """기존 최적화 결과 로드"""
    file_path = f"{CFG['OUTPUT_DIR']}/model_optimization/{model_name.lower()}_optuna.json"
    
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            log_message(f"    기존 최적화 결과 로드: {file_path}")
            return results
        except Exception as e:
            log_message(f"    기존 결과 로드 실패: {str(e)}")
            return None
    
    return None

def optimize_model(model_name, X_train, y_train, X_val, y_val, y_ic50_val):
    """Optuna를 사용한 하이퍼파라미터 최적화 (기존 결과 재사용)"""
    
    # 기존 결과 확인
    existing_results = load_existing_optimization_results(model_name)
    if existing_results and len(existing_results) >= CFG['N_FOLDS']:
        log_message(f"    기존 최적화 결과 사용")
        return existing_results[0], None  # 첫 번째 fold의 결과 사용
    
    def objective(trial):
        try:
            if model_name == 'RandomForest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': CFG['SEED']
                }
                model = RandomForestRegressor(**params)
                
            elif model_name == 'ExtraTrees':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': CFG['SEED']
                }
                model = ExtraTreesRegressor(**params)
                
            elif model_name == 'LightGBM':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'num_leaves': trial.suggest_int('num_leaves', 15, 100),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': CFG['SEED'],
                    'verbose': -1
                }
                model = lgb.LGBMRegressor(**params)
                
            elif model_name == 'XGBoost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': CFG['SEED'],
                    'verbosity': 0
                }
                model = xgb.XGBRegressor(**params)
                
            elif model_name == 'CatBoost':
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'random_state': CFG['SEED'],
                    'verbose': False,
                    'allow_writing_files': False
                }
                model = cb.CatBoostRegressor(**params)
            
            # 모델 학습
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_val)
            y_ic50_pred = pIC50_to_IC50(y_pred)
            
            # 점수 계산
            normalized_rmse = calculate_normalized_rmse(y_ic50_val, y_ic50_pred)
            correlation_squared = calculate_correlation(y_val, y_pred)
            final_score = calculate_final_score(normalized_rmse, correlation_squared)
            
            return final_score
            
        except Exception as e:
            print(f"Trial failed: {str(e)}")
            return 0.0
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=CFG['OPTUNA_TRIALS'])
    
    return study.best_params, study.best_value

def train_and_evaluate_models(X, y, y_ic50):
    """모든 모델 학습 및 평가 (기존 결과 재사용)"""
    
    models = ['RandomForest', 'ExtraTrees', 'LightGBM', 'XGBoost', 'CatBoost']
    kf = KFold(n_splits=CFG['N_FOLDS'], shuffle=True, random_state=CFG['SEED'])
    
    results = {}
    cv_predictions = {}
    
    log_message("=== 모델 학습 및 평가 시작 ===")
    
    for model_name in models:
        log_message(f"모델 {model_name} 처리 중...")
        
        # 기존 최적화 결과 확인
        existing_results = load_existing_optimization_results(model_name)
        use_existing = existing_results and len(existing_results) >= CFG['N_FOLDS']
        
        if use_existing:
            log_message(f"  기존 최적화 결과 사용 ({len(existing_results)}개 fold)")
        
        model_scores = []
        fold_predictions = []
        best_params_list = existing_results if use_existing else []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            log_message(f"  Fold {fold + 1}/{CFG['N_FOLDS']}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            y_ic50_train, y_ic50_val = y_ic50[train_idx], y_ic50[val_idx]
            
            # 하이퍼파라미터 가져오기
            try:
                if use_existing and fold < len(existing_results):
                    best_params = existing_results[fold]
                    log_message(f"    기존 파라미터 사용: fold {fold}")
                else:
                    best_params, best_score = optimize_model(model_name, X_train, y_train, X_val, y_val, y_ic50_val)
                    if not use_existing:
                        best_params_list.append(best_params)
                
                # 최적 파라미터로 모델 재학습
                if model_name == 'RandomForest':
                    model = RandomForestRegressor(**best_params)
                elif model_name == 'ExtraTrees':
                    model = ExtraTreesRegressor(**best_params)
                elif model_name == 'LightGBM':
                    model = lgb.LGBMRegressor(**best_params)
                elif model_name == 'XGBoost':
                    model = xgb.XGBRegressor(**best_params)
                elif model_name == 'CatBoost':
                    model = cb.CatBoostRegressor(**best_params)
                
                model.fit(X_train, y_train)
                
                # 예측
                y_pred = model.predict(X_val)
                y_ic50_pred = pIC50_to_IC50(y_pred)
                
                # 점수 계산
                normalized_rmse = calculate_normalized_rmse(y_ic50_val, y_ic50_pred)
                correlation_squared = calculate_correlation(y_val, y_pred)
                final_score = calculate_final_score(normalized_rmse, correlation_squared)
                
                model_scores.append(final_score)
                fold_predictions.append({
                    'fold': fold,
                    'val_idx': val_idx,
                    'y_true': y_val,
                    'y_pred': y_pred,
                    'y_ic50_true': y_ic50_val,
                    'y_ic50_pred': y_ic50_pred,
                    'score': final_score
                })
                
                log_message(f"    Score: {final_score:.4f}")
                
            except Exception as e:
                log_message(f"    오류 발생: {str(e)}")
                model_scores.append(0.0)
                fold_predictions.append(None)
        
        # 모델 결과 저장
        results[model_name] = {
            'cv_scores': model_scores,
            'mean_score': np.mean(model_scores),
            'std_score': np.std(model_scores),
            'best_params': best_params_list
        }
        
        cv_predictions[model_name] = fold_predictions
        
        # 새로운 최적화 결과만 저장
        if not use_existing:
            with open(f"{CFG['OUTPUT_DIR']}/model_optimization/{model_name.lower()}_optuna.json", 'w') as f:
                json.dump(best_params_list, f, indent=2)
        
        log_message(f"  {model_name} 완료 - 평균 점수: {np.mean(model_scores):.4f} ± {np.std(model_scores):.4f}")
    
    return results, cv_predictions

def create_ensembles(results, cv_predictions, X, y):
    """앙상블 모델 생성"""
    
    log_message("=== 앙상블 모델 생성 ===")
    
    # 상위 3개 모델 선택
    sorted_models = sorted(results.items(), key=lambda x: x[1]['mean_score'], reverse=True)
    top3_models = [model_name for model_name, _ in sorted_models[:3]]
    
    log_message(f"상위 3개 모델: {top3_models}")
    
    # 1단 앙상블: 가중 평균
    ensemble_predictions = []
    weights = []
    
    for model_name in top3_models:
        weights.append(results[model_name]['mean_score'])
    
    weights = np.array(weights)
    weights = weights / np.sum(weights)  # 정규화
    
    log_message(f"앙상블 가중치: {dict(zip(top3_models, weights))}")
    
    # CV 예측 결합
    ensemble_scores = []
    for fold in range(CFG['N_FOLDS']):
        fold_preds = []
        y_true = None
        y_ic50_true = None
        
        for i, model_name in enumerate(top3_models):
            if cv_predictions[model_name][fold] is not None:
                fold_preds.append(cv_predictions[model_name][fold]['y_pred'])
                if y_true is None:
                    y_true = cv_predictions[model_name][fold]['y_true']
                    y_ic50_true = cv_predictions[model_name][fold]['y_ic50_true']
        
        if len(fold_preds) == len(top3_models):
            # 가중 평균
            ensemble_pred = np.average(fold_preds, axis=0, weights=weights)
            ensemble_ic50_pred = pIC50_to_IC50(ensemble_pred)
            
            # 점수 계산
            normalized_rmse = calculate_normalized_rmse(y_ic50_true, ensemble_ic50_pred)
            correlation_squared = calculate_correlation(y_true, ensemble_pred)
            final_score = calculate_final_score(normalized_rmse, correlation_squared)
            
            ensemble_scores.append(final_score)
    
    ensemble_result = {
        'model_names': top3_models,
        'weights': weights.tolist(),
        'cv_scores': ensemble_scores,
        'mean_score': np.mean(ensemble_scores),
        'std_score': np.std(ensemble_scores)
    }
    
    log_message(f"앙상블 점수: {np.mean(ensemble_scores):.4f} ± {np.std(ensemble_scores):.4f}")
    
    return ensemble_result

def select_best_model(results, ensemble_result):
    """최고 성능 모델 선택"""
    
    all_models = dict(results)
    all_models['Ensemble'] = ensemble_result
    
    best_model = max(all_models.items(), key=lambda x: x[1]['mean_score'])
    best_name, best_info = best_model
    
    log_message(f"최고 성능 모델: {best_name} (점수: {best_info['mean_score']:.4f})")
    
    return best_name, best_info

def train_final_model(best_model_name, X, y, results):
    """최종 모델 학습"""
    
    log_message(f"최종 모델 {best_model_name} 학습 중...")
    
    if best_model_name == 'Ensemble':
        # 앙상블의 경우 각 모델을 다시 학습
        models = []
        for model_name in results['Ensemble']['model_names']:
            # 최고 파라미터 사용 (첫 번째 폴드의 파라미터)
            best_params = results[model_name]['best_params'][0]
            
            if model_name == 'RandomForest':
                model = RandomForestRegressor(**best_params)
            elif model_name == 'ExtraTrees':
                model = ExtraTreesRegressor(**best_params)
            elif model_name == 'LightGBM':
                model = lgb.LGBMRegressor(**best_params)
            elif model_name == 'XGBoost':
                model = xgb.XGBRegressor(**best_params)
            elif model_name == 'CatBoost':
                model = cb.CatBoostRegressor(**best_params)
            
            model.fit(X, y)
            models.append(model)
        
        return models, results['Ensemble']['weights']
    
    else:
        # 단일 모델의 경우
        best_params = results[best_model_name]['best_params'][0]
        
        if best_model_name == 'RandomForest':
            model = RandomForestRegressor(**best_params)
        elif best_model_name == 'ExtraTrees':
            model = ExtraTreesRegressor(**best_params)
        elif best_model_name == 'LightGBM':
            model = lgb.LGBMRegressor(**best_params)
        elif best_model_name == 'XGBoost':
            model = xgb.XGBRegressor(**best_params)
        elif best_model_name == 'CatBoost':
            model = cb.CatBoostRegressor(**best_params)
        
        model.fit(X, y)
        return model, None

def make_predictions(final_model, weights, scaler):
    """테스트 데이터 예측"""
    
    log_message("테스트 데이터 예측 중...")
    
    # 테스트 데이터 로드
    test_df = pd.read_csv("data/test.csv")
    
    # 컬럼명 확인 및 수정
    log_message(f"테스트 데이터 컬럼: {test_df.columns.tolist()}")
    
    # 가능한 SMILES 컬럼명들
    smiles_columns = ['Smiles', 'SMILES', 'smiles']
    smiles_col = None
    
    for col in smiles_columns:
        if col in test_df.columns:
            smiles_col = col
            break
    
    if smiles_col is None:
        raise ValueError(f"SMILES 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {test_df.columns.tolist()}")
    
    log_message(f"SMILES 컬럼 사용: {smiles_col}")
    
    # 특성 계산
    test_mol_props = test_df[smiles_col].apply(calculate_molecular_properties)
    test_mol_props_df = pd.DataFrame(test_mol_props.tolist())
    
    # Morgan fingerprint
    test_df['Fingerprint'] = test_df[smiles_col].apply(smiles_to_fingerprint)
    test_df = test_df[test_df['Fingerprint'].notnull()]
    
    # 피처 매트릭스 생성
    X_test_fp = np.stack(test_df['Fingerprint'].values)
    
    # 분자 특성이 테스트 데이터와 일치하는지 확인
    if len(test_mol_props_df) != len(test_df):
        # 인덱스 맞추기
        test_mol_props_df = test_mol_props_df.iloc[:len(test_df)].reset_index(drop=True)
    
    X_test_props = scaler.transform(test_mol_props_df)
    X_test = np.hstack([X_test_fp, X_test_props])
    
    log_message(f"테스트 피처 매트릭스 크기: {X_test.shape}")
    
    # 예측
    if isinstance(final_model, list):  # 앙상블
        predictions = []
        for model in final_model:
            pred = model.predict(X_test)
            predictions.append(pred)
        
        # 가중 평균
        final_predictions = np.average(predictions, axis=0, weights=weights)
    else:  # 단일 모델
        final_predictions = final_model.predict(X_test)
    
    # IC50로 변환
    test_ic50_predictions = pIC50_to_IC50(final_predictions)
    
    # 제출 파일 생성
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'ASK1_IC50_nM': test_ic50_predictions
    })
    
    submission.to_csv(f"{CFG['OUTPUT_DIR']}/submission.csv", index=False)
    
    log_message(f"예측 완료: {len(submission)}개 샘플")
    
    return submission

def save_results(results, ensemble_result, best_model_name, best_info):
    """결과 저장"""
    
    # 개별 모델 결과
    model_results = []
    for model_name, info in results.items():
        model_results.append({
            'model': model_name,
            'mean_score': info['mean_score'],
            'std_score': info['std_score'],
            'cv_scores': info['cv_scores']
        })
    
    # 앙상블 결과 추가
    if ensemble_result:
        model_results.append({
            'model': 'Ensemble',
            'mean_score': ensemble_result['mean_score'],
            'std_score': ensemble_result['std_score'],
            'cv_scores': ensemble_result['cv_scores']
        })
    
    results_df = pd.DataFrame(model_results)
    results_df.to_csv(f"{CFG['OUTPUT_DIR']}/cv_results/individual_models.csv", index=False)
    
    # 전체 설정 저장
    config = {
        'experiment_name': CFG['EXPERIMENT_NAME'],
        'timestamp': datetime.now().isoformat(),
        'config': CFG,
        'best_model': best_model_name,
        'best_score': best_info['mean_score'],
        'results_summary': model_results
    }
    
    with open(f"{CFG['OUTPUT_DIR']}/config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # 실험 요약
    summary = f"""
=== Try6 실험 요약 ===
실험 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
최고 모델: {best_model_name}
최고 점수: {best_info['mean_score']:.4f} ± {best_info['std_score']:.4f}

개별 모델 성능:
"""
    
    for result in model_results:
        summary += f"- {result['model']}: {result['mean_score']:.4f} ± {result['std_score']:.4f}\n"
    
    with open(f"{CFG['OUTPUT_DIR']}/experiment_summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary)
    
    log_message("결과 저장 완료")

def main():
    """메인 실행 함수"""
    
    try:
        # 초기 설정
        seed_everything(CFG['SEED'])
        setup_output_directory()
        
        log_message("=== Try6 실험 시작 ===")
        log_message(f"설정: {CFG}")
        
        # 1. 데이터 전처리
        df, mol_props_df = load_and_preprocess_data()
        
        # 2. 피처 매트릭스 생성
        X, scaler = create_feature_matrix(df, mol_props_df)
        y = df['pIC50'].values
        y_ic50 = df['ic50_nM'].values
        
        log_message(f"피처 매트릭스 크기: {X.shape}")
        
        # 3. 모델 학습 및 평가
        results, cv_predictions = train_and_evaluate_models(X, y, y_ic50)
        
        # 4. 앙상블 생성
        ensemble_result = create_ensembles(results, cv_predictions, X, y)
        
        # 5. 최고 모델 선택
        best_model_name, best_info = select_best_model(results, ensemble_result)
        
        # 6. 최종 모델 학습
        final_model, weights = train_final_model(best_model_name, X, y, 
                                               {**results, 'Ensemble': ensemble_result})
        
        # 7. 예측 및 제출
        submission = make_predictions(final_model, weights, scaler)
        
        # 8. 결과 저장
        save_results(results, ensemble_result, best_model_name, best_info)
        
        # 9. 모델 저장
        if isinstance(final_model, list):
            for i, model in enumerate(final_model):
                with open(f"{CFG['OUTPUT_DIR']}/models/ensemble_model_{i}.pkl", 'wb') as f:
                    pickle.dump(model, f)
            with open(f"{CFG['OUTPUT_DIR']}/models/ensemble_weights.pkl", 'wb') as f:
                pickle.dump(weights, f)
        else:
            with open(f"{CFG['OUTPUT_DIR']}/models/best_model.pkl", 'wb') as f:
                pickle.dump(final_model, f)
        
        # 스케일러 저장
        with open(f"{CFG['OUTPUT_DIR']}/models/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        log_message("=== Try6 실험 완료 ===")
        log_message(f"최종 결과: {best_model_name} - {best_info['mean_score']:.4f} ± {best_info['std_score']:.4f}")
        
    except Exception as e:
        error_msg = f"실험 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        log_message(error_msg)
        
        # 오류 로그 저장
        with open(f"{CFG['OUTPUT_DIR']}/error_log.txt", 'w', encoding='utf-8') as f:
            f.write(error_msg)
        
        raise

if __name__ == "__main__":
    main() 