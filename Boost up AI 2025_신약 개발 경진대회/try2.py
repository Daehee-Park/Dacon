import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 커스텀 메트릭 import
from custom_metrics import competition_scorer

# RDKit imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdFingerprintGenerator
    from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumHBD, CalcNumHBA
    from rdkit.Chem import SaltRemover
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    rdkit_available = True
except ImportError:
    print("RDKit not available. Please install: pip install rdkit")
    rdkit_available = False

# ML imports
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna
import joblib
import json

def setup_logging(output_dir):
    """로깅 설정"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f'experiment_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_and_standardize_smiles(smiles):
    """SMILES 검증 및 표준화 (간소화)"""
    if not rdkit_available:
        return smiles
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 염 제거
        remover = SaltRemover.SaltRemover()
        mol_clean = remover.StripMol(mol)
        
        # Canonical SMILES 변환
        return Chem.MolToSmiles(mol_clean, canonical=True)
    except:
        return None

def calculate_core_properties(smiles):
    """핵심 분자 특성만 계산 (5-10개)"""
    if not rdkit_available:
        return {}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        # CYP3A4 관련 핵심 특성만 선별
        properties = {
            'MW': Descriptors.MolWt(mol),           # 분자량
            'LogP': Crippen.MolLogP(mol),           # 소수성 (중요)
            'tPSA': CalcTPSA(mol),                  # 극성 표면적
            'HBD': CalcNumHBD(mol),                 # 수소결합 공여체
            'HBA': CalcNumHBA(mol),                 # 수소결합 수용체
            'NumRotBonds': rdMolDescriptors.CalcNumRotatableBonds(mol),  # 회전 가능한 결합
            'NumAromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),  # 방향족 고리
            'FractionCsp3': rdMolDescriptors.CalcFractionCsp3(mol),  # sp3 탄소 비율
        }
        
        return properties
    except:
        return {}

def generate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Morgan fingerprint 생성 (단일 지문만)"""
    if not rdkit_available:
        return {}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        # Morgan Fingerprint만 사용
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        morgan_fp = morgan_gen.GetFingerprint(mol)
        morgan_bits = {f'Morgan_{i}': int(morgan_fp.GetBit(i)) for i in range(n_bits)}
        
        return morgan_bits
    except:
        return {}

def experiment_morgan_parameters(smiles_list, logger):
    """Morgan fingerprint 파라미터 실험"""
    logger.info("Experimenting with Morgan fingerprint parameters...")
    
    parameter_sets = [
        {'radius': 1, 'n_bits': 1024},
        {'radius': 2, 'n_bits': 1024},
        {'radius': 3, 'n_bits': 1024},
        {'radius': 2, 'n_bits': 2048},
        {'radius': 3, 'n_bits': 2048},
        {'radius': 2, 'n_bits': 4096},
    ]
    
    fingerprint_datasets = {}
    
    for params in parameter_sets:
        param_name = f"r{params['radius']}_b{params['n_bits']}"
        logger.info(f"Generating {param_name} fingerprints...")
        
        fps_list = []
        for smiles in smiles_list:
            fps = generate_morgan_fingerprint(smiles, **params)
            fps_list.append(fps)
        
        fingerprint_datasets[param_name] = pd.DataFrame(fps_list)
        logger.info(f"{param_name}: {fingerprint_datasets[param_name].shape[1]} features")
    
    return fingerprint_datasets

def apply_target_transformations(y_train, logger):
    """타겟 변환 실험"""
    logger.info("Applying target transformations...")
    
    transformations = {}
    
    # 1. 원본
    transformations['original'] = y_train.copy()
    
    # 2. Log transformation
    transformations['log'] = pd.Series(np.log1p(y_train), index=y_train.index)
    
    # 3. Quantile transformation (uniform)
    qt_uniform = QuantileTransformer(output_distribution='uniform', random_state=42)
    transformed_uniform = qt_uniform.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    transformations['quantile_uniform'] = pd.Series(transformed_uniform, index=y_train.index)
    
    # 4. Quantile transformation (normal)
    qt_normal = QuantileTransformer(output_distribution='normal', random_state=42)
    transformed_normal = qt_normal.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    transformations['quantile_normal'] = pd.Series(transformed_normal, index=y_train.index)
    
    # 5. Power transformation (Yeo-Johnson)
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    transformed_power = pt.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    transformations['power'] = pd.Series(transformed_power, index=y_train.index)
    
    # 변환기 저장
    transformers = {
        'quantile_uniform': qt_uniform,
        'quantile_normal': qt_normal,
        'power': pt
    }
    
    for name, y_trans in transformations.items():
        logger.info(f"Target '{name}': mean={y_trans.mean():.4f}, std={y_trans.std():.4f}")
    
    return transformations, transformers

def calculate_score(y_true, y_pred):
    """경진대회 평가 점수 계산"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_range = y_true.max() - y_true.min()
    normalized_rmse = rmse / y_range
    A = normalized_rmse
    
    correlation, _ = pearsonr(y_true, y_pred)
    B = correlation if not np.isnan(correlation) else 0
    
    score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    return score, A, B

def optimize_lightgbm(X_train, y_train, cv_folds, logger):
    """LightGBM 하이퍼파라미터 최적화"""
    logger.info("Optimizing LightGBM parameters...")
    
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'seed': 42,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }
        
        cv_scores = []
        for train_idx, val_idx in cv_folds:
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            
            score, _, _ = calculate_score(y_fold_val, y_pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    return study.best_params

def optimize_xgboost(X_train, y_train, cv_folds, logger):
    """XGBoost 하이퍼파라미터 최적화"""
    logger.info("Optimizing XGBoost parameters...")
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'verbosity': 0,
            'seed': 42,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        }
        
        cv_scores = []
        for train_idx, val_idx in cv_folds:
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            
            score, _, _ = calculate_score(y_fold_val, y_pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    return study.best_params

def optimize_catboost(X_train, y_train, cv_folds, logger):
    """CatBoost 하이퍼파라미터 최적화"""
    logger.info("Optimizing CatBoost parameters...")
    
    def objective(trial):
        params = {
            'loss_function': 'RMSE',
            'verbose': False,
            'random_seed': 42,
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
        }
        
        cv_scores = []
        for train_idx, val_idx in cv_folds:
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = cb.CatBoostRegressor(**params)
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            
            score, _, _ = calculate_score(y_fold_val, y_pred)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    return study.best_params

def train_ensemble_models(X_train, y_train, cv_folds, best_params, logger):
    """앙상블 모델 학습"""
    logger.info("Training ensemble models...")
    
    models = {}
    cv_results = {}
    
    # LightGBM
    lgb_scores = []
    lgb_models = []
    for train_idx, val_idx in cv_folds:
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**best_params['lgb'])
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        
        score, _, _ = calculate_score(y_fold_val, y_pred)
        lgb_scores.append(score)
        lgb_models.append(model)
    
    models['lgb'] = lgb_models
    cv_results['lgb'] = np.mean(lgb_scores)
    
    # XGBoost
    xgb_scores = []
    xgb_models = []
    for train_idx, val_idx in cv_folds:
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBRegressor(**best_params['xgb'])
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        
        score, _, _ = calculate_score(y_fold_val, y_pred)
        xgb_scores.append(score)
        xgb_models.append(model)
    
    models['xgb'] = xgb_models
    cv_results['xgb'] = np.mean(xgb_scores)
    
    # CatBoost
    cb_scores = []
    cb_models = []
    for train_idx, val_idx in cv_folds:
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = cb.CatBoostRegressor(**best_params['cb'])
        model.fit(X_fold_train, y_fold_train)
        y_pred = model.predict(X_fold_val)
        
        score, _, _ = calculate_score(y_fold_val, y_pred)
        cb_scores.append(score)
        cb_models.append(model)
    
    models['cb'] = cb_models
    cv_results['cb'] = np.mean(cb_scores)
    
    logger.info(f"LightGBM CV Score: {cv_results['lgb']:.4f}")
    logger.info(f"XGBoost CV Score: {cv_results['xgb']:.4f}")
    logger.info(f"CatBoost CV Score: {cv_results['cb']:.4f}")
    
    return models, cv_results

def main():
    output_dir = "output/try2"
    data_dir = "data"
    
    logger = setup_logging(output_dir)
    logger.info("Starting try2: Simplified and optimized approach")
    
    if not rdkit_available:
        logger.error("RDKit not available. Exiting.")
        return
    
    try:
        # 데이터 로드
        logger.info("Loading data...")
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        logger.info(f"Train: {train_df.shape}, Test: {test_df.shape}")
        
        # 간단한 SMILES 정제
        logger.info("Standardizing SMILES...")
        train_df['Clean_SMILES'] = train_df['Canonical_Smiles'].apply(validate_and_standardize_smiles)
        test_df['Clean_SMILES'] = test_df['Canonical_Smiles'].apply(validate_and_standardize_smiles)
        
        # 유효하지 않은 SMILES 제거
        train_df = train_df.dropna(subset=['Clean_SMILES'])
        test_df = test_df.dropna(subset=['Clean_SMILES'])
        logger.info(f"After cleaning - Train: {train_df.shape}, Test: {test_df.shape}")
        
        # Morgan fingerprint 파라미터 실험
        all_smiles = list(train_df['Clean_SMILES']) + list(test_df['Clean_SMILES'])
        fingerprint_datasets = experiment_morgan_parameters(all_smiles, logger)
        
        # 핵심 분자 특성 계산
        logger.info("Calculating core molecular properties...")
        core_props_list = []
        for smiles in all_smiles:
            props = calculate_core_properties(smiles)
            core_props_list.append(props)
        
        core_props_df = pd.DataFrame(core_props_list)
        logger.info(f"Core properties: {core_props_df.shape[1]} features")
        
        # 타겟 변환 실험
        y_train = train_df['Inhibition']
        target_transformations, transformers = apply_target_transformations(y_train, logger)
        
        # 각 조합별 실험
        best_config = None
        best_score = -np.inf
        results = []
        
        for fp_name, fp_df in fingerprint_datasets.items():
            # 특성 결합
            train_fps = fp_df[:len(train_df)]
            test_fps = fp_df[len(train_df):]
            
            train_core_props = core_props_df[:len(train_df)]
            test_core_props = core_props_df[len(train_df):]
            
            # 특성 결합
            X_train_combined = pd.concat([train_fps, train_core_props], axis=1).fillna(0)
            X_test_combined = pd.concat([test_fps, test_core_props], axis=1).fillna(0)
            
            # 분산 기반 특성 선택
            from sklearn.feature_selection import VarianceThreshold
            var_selector = VarianceThreshold(threshold=0.01)
            X_train_selected = var_selector.fit_transform(X_train_combined)
            X_test_selected = var_selector.transform(X_test_combined)
            
            selected_features = X_train_combined.columns[var_selector.get_support()]
            X_train_df = pd.DataFrame(X_train_selected, columns=selected_features)
            X_test_df = pd.DataFrame(X_test_selected, columns=selected_features)
            
            logger.info(f"Fingerprint: {fp_name}, Features: {len(selected_features)}")
            
            # 각 타겟 변환별 실험
            for target_name, y_transformed in target_transformations.items():
                logger.info(f"Testing {fp_name} + {target_name}")
                
                # 스케일링
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_df)
                X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=selected_features)
                
                # 교차 검증 설정
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                cv_folds = list(kf.split(X_train_scaled_df))
                
                # 하이퍼파라미터 최적화
                best_params = {}
                best_params['lgb'] = optimize_lightgbm(X_train_scaled_df, y_transformed, cv_folds, logger)
                best_params['xgb'] = optimize_xgboost(X_train_scaled_df, y_transformed, cv_folds, logger)
                best_params['cb'] = optimize_catboost(X_train_scaled_df, y_transformed, cv_folds, logger)
                
                # 앙상블 모델 학습
                models, cv_results = train_ensemble_models(X_train_scaled_df, y_transformed, cv_folds, best_params, logger)
                
                # 최고 성능 모델 선택
                best_model_name = max(cv_results, key=cv_results.get)
                config_score = cv_results[best_model_name]
                
                result = {
                    'fingerprint': fp_name,
                    'target_transform': target_name,
                    'best_model': best_model_name,
                    'cv_score': config_score,
                    'feature_count': len(selected_features)
                }
                results.append(result)
                
                logger.info(f"Config: {fp_name} + {target_name} + {best_model_name} = {config_score:.4f}")
                
                # 최고 성능 업데이트
                if config_score > best_score:
                    best_score = config_score
                    best_config = {
                        'fingerprint': fp_name,
                        'target_transform': target_name,
                        'best_model': best_model_name,
                        'models': models,
                        'cv_results': cv_results,
                        'best_params': best_params,
                        'X_train': X_train_scaled_df,
                        'X_test': X_test_df,
                        'y_train': y_transformed,
                        'scaler': scaler,
                        'var_selector': var_selector,
                        'transformers': transformers,
                        'selected_features': selected_features
                    }
        
        # 최고 설정으로 최종 예측
        logger.info(f"Best configuration: {best_config['fingerprint']} + {best_config['target_transform']} + {best_config['best_model']} (Score: {best_score:.4f})")
        
        # 테스트 데이터 스케일링
        X_test_scaled = best_config['scaler'].transform(best_config['X_test'])
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=best_config['selected_features'])
        
        # 최종 예측 (앙상블)
        test_predictions = []
        for model in best_config['models'][best_config['best_model']]:
            pred = model.predict(X_test_scaled_df)
            test_predictions.append(pred)
        
        # 평균 앙상블
        final_predictions = np.mean(test_predictions, axis=0)
        
        # 타겟 역변환
        if best_config['target_transform'] == 'log':
            final_predictions = np.expm1(final_predictions)
        elif best_config['target_transform'] == 'quantile_uniform':
            final_predictions = best_config['transformers']['quantile_uniform'].inverse_transform(final_predictions.reshape(-1, 1)).flatten()
        elif best_config['target_transform'] == 'quantile_normal':
            final_predictions = best_config['transformers']['quantile_normal'].inverse_transform(final_predictions.reshape(-1, 1)).flatten()
        elif best_config['target_transform'] == 'power':
            final_predictions = best_config['transformers']['power'].inverse_transform(final_predictions.reshape(-1, 1)).flatten()
        
        # 예측값 범위 조정
        final_predictions = np.clip(final_predictions, 0, 100)
        
        # 제출 파일 생성
        submission_df = pd.DataFrame({
            'ID': test_df['ID'],
            'Inhibition': final_predictions
        })
        
        submission_path = os.path.join(output_dir, 'submission.csv')
        submission_df.to_csv(submission_path, index=False)
        logger.info(f"Submission saved: {submission_path}")
        
        # 결과 요약 저장
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(output_dir, 'all_configurations.csv'), index=False)
        
        config_summary = {
            'best_fingerprint': best_config['fingerprint'],
            'best_target_transform': best_config['target_transform'],
            'best_model': best_config['best_model'],
            'best_cv_score': float(best_score),
            'feature_count': len(best_config['selected_features']),
            'all_results': results
        }
        
        with open(os.path.join(output_dir, 'config_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(config_summary, f, indent=2, ensure_ascii=False)
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Best score: {best_score:.4f}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 