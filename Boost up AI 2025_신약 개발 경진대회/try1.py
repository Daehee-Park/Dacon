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
    from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcTPSA, CalcNumHBD, CalcNumHBA
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.Chem import SaltRemover
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    rdkit_available = True
except ImportError:
    print("RDKit not available. Please install: pip install rdkit")
    rdkit_available = False

# AutoGluon import
try:
    from autogluon.tabular import TabularPredictor
    autogluon_available = True
except ImportError:
    print("AutoGluon not available. Please install: pip install autogluon")
    autogluon_available = False

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import joblib

def setup_logging(output_dir):
    """로깅 설정"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 로그 파일 경로
    log_file = os.path.join(output_dir, f'experiment_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_smiles(smiles):
    """SMILES 검증 및 정규화"""
    if not rdkit_available:
        return smiles
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Canonical SMILES로 변환
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        return canonical_smiles
    except:
        return None

def remove_salts_and_standardize(smiles):
    """염 제거 및 표준화"""
    if not rdkit_available:
        return smiles
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # 염 제거
        remover = SaltRemover.SaltRemover()
        mol_no_salt = remover.StripMol(mol)
        
        # 표준화된 SMILES 반환
        return Chem.MolToSmiles(mol_no_salt, canonical=True)
    except:
        return None

def calculate_molecular_properties(smiles):
    """기본 분자 특성 계산"""
    if not rdkit_available:
        return {}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        properties = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'tPSA': CalcTPSA(mol),
            'RotatableBonds': CalcNumRotatableBonds(mol),
            'HBD': CalcNumHBD(mol),
            'HBA': CalcNumHBA(mol),
            'NumAtoms': mol.GetNumAtoms(),
            'NumBonds': mol.GetNumBonds(),
            'NumRings': rdMolDescriptors.CalcNumRings(mol),
            'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
            'SaturatedRings': rdMolDescriptors.CalcNumSaturatedRings(mol),
            'HeteroAtoms': rdMolDescriptors.CalcNumHeteroatoms(mol),
            'FractionCsp3': rdMolDescriptors.CalcFractionCsp3(mol),
            'MolMR': Crippen.MolMR(mol),
        }
        
        return properties
    except:
        return {}

def generate_fingerprints(smiles, radius=2, n_bits=2048):
    """분자 지문 생성"""
    if not rdkit_available:
        return {}
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        # Morgan Fingerprint
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        morgan_fp = morgan_gen.GetFingerprint(mol)
        morgan_bits = {f'Morgan_{i}': int(morgan_fp.GetBit(i)) for i in range(n_bits)}
        
        # MACCS Keys
        from rdkit.Chem import MACCSkeys
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_bits = {f'MACCS_{i}': int(maccs_fp.GetBit(i)) for i in range(maccs_fp.GetNumBits())}
        
        # RDKit Fingerprint
        rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
        rdkit_fp = rdkit_gen.GetFingerprint(mol)
        rdkit_bits = {f'RDKit_{i}': int(rdkit_fp.GetBit(i)) for i in range(n_bits)}
        
        fingerprints = {}
        fingerprints.update(morgan_bits)
        fingerprints.update(maccs_bits)
        fingerprints.update(rdkit_bits)
        
        return fingerprints
    except:
        return {}

def process_data(df, logger, is_train=True):
    """데이터 전처리 파이프라인"""
    logger.info(f"Processing {'train' if is_train else 'test'} data...")
    logger.info(f"Initial shape: {df.shape}")
    
    # 1. SMILES 검증 및 정규화
    logger.info("Step 1: SMILES validation and canonicalization")
    df['Valid_SMILES'] = df['Canonical_Smiles'].apply(validate_smiles)
    invalid_count = df['Valid_SMILES'].isna().sum()
    logger.info(f"Invalid SMILES found: {invalid_count}")
    
    if invalid_count > 0:
        df = df.dropna(subset=['Valid_SMILES'])
        logger.info(f"After removing invalid SMILES: {df.shape}")
    
    # 2. 염 제거 및 표준화
    logger.info("Step 2: Salt removal and standardization")
    df['Standardized_SMILES'] = df['Valid_SMILES'].apply(remove_salts_and_standardize)
    df = df.dropna(subset=['Standardized_SMILES'])
    logger.info(f"After standardization: {df.shape}")
    
    # 3. 중복 제거
    if is_train:
        logger.info("Step 3: Removing duplicates")
        initial_count = len(df)
        df = df.drop_duplicates(subset=['Standardized_SMILES'])
        logger.info(f"Duplicates removed: {initial_count - len(df)}")
        logger.info(f"After duplicate removal: {df.shape}")
    
    # 4. 분자 특성 계산
    logger.info("Step 4: Calculating molecular properties")
    properties_list = []
    for smiles in df['Standardized_SMILES']:
        props = calculate_molecular_properties(smiles)
        properties_list.append(props)
    
    properties_df = pd.DataFrame(properties_list)
    logger.info(f"Molecular properties calculated: {properties_df.shape[1]} features")
    
    # 5. 지문 생성
    logger.info("Step 5: Generating fingerprints")
    fingerprints_list = []
    for smiles in df['Standardized_SMILES']:
        fps = generate_fingerprints(smiles)
        fingerprints_list.append(fps)
    
    fingerprints_df = pd.DataFrame(fingerprints_list)
    logger.info(f"Fingerprints generated: {fingerprints_df.shape[1]} features")
    
    # 6. 특성 결합
    logger.info("Step 6: Combining features")
    df_reset = df.reset_index(drop=True)
    properties_df_reset = properties_df.reset_index(drop=True)
    fingerprints_df_reset = fingerprints_df.reset_index(drop=True)
    
    combined_df = pd.concat([df_reset, properties_df_reset, fingerprints_df_reset], axis=1)
    logger.info(f"Combined features shape: {combined_df.shape}")
    
    return combined_df

def select_features(X_train, y_train, logger, variance_threshold=0.01):
    """특성 선택"""
    logger.info("Starting feature selection...")
    logger.info(f"Initial features: {X_train.shape[1]}")
    
    # 1. Variance Threshold
    variance_selector = VarianceThreshold(threshold=variance_threshold)
    X_train_var = variance_selector.fit_transform(X_train)
    
    selected_features = X_train.columns[variance_selector.get_support()]
    logger.info(f"After variance threshold ({variance_threshold}): {len(selected_features)} features")
    
    # 2. 상관계수 기반 중복 제거
    X_train_var_df = pd.DataFrame(X_train_var, columns=selected_features)
    corr_matrix = X_train_var_df.corr().abs()
    
    # 상관계수가 높은 특성 쌍 찾기
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.95:  # 0.95 이상 상관계수
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
    
    # 중복 특성 제거
    features_to_remove = set()
    for feat1, feat2 in high_corr_pairs:
        features_to_remove.add(feat2)  # 두 번째 특성 제거
    
    final_features = [f for f in selected_features if f not in features_to_remove]
    logger.info(f"After correlation removal: {len(final_features)} features")
    
    return final_features, variance_selector

def calculate_score(y_true, y_pred):
    """경진대회 평가 점수 계산"""
    # A: Normalized RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_range = y_true.max() - y_true.min()
    normalized_rmse = rmse / y_range
    A = normalized_rmse
    
    # B: Pearson correlation
    correlation, _ = pearsonr(y_true, y_pred)
    B = correlation if not np.isnan(correlation) else 0
    
    # Final score
    score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    
    return score, A, B

def main():
    # 설정
    output_dir = "output/try1"
    data_dir = "data"
    
    # 로깅 설정
    logger = setup_logging(output_dir)
    logger.info("Starting CYP3A4 inhibition prediction experiment")
    logger.info(f"RDKit available: {rdkit_available}")
    logger.info(f"AutoGluon available: {autogluon_available}")
    
    if not rdkit_available or not autogluon_available:
        logger.error("Required libraries not available. Exiting.")
        return
    
    try:
        # 데이터 로드
        logger.info("Loading data...")
        train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))
        logger.info(f"Train shape: {train_df.shape}")
        logger.info(f"Test shape: {test_df.shape}")
        
        # 데이터 전처리
        train_processed = process_data(train_df, logger, is_train=True)
        test_processed = process_data(test_df, logger, is_train=False)
        
        # 특성과 타겟 분리
        feature_columns = [col for col in train_processed.columns 
                          if col not in ['ID', 'Canonical_Smiles', 'Valid_SMILES', 
                                       'Standardized_SMILES', 'Inhibition']]
        
        X_train = train_processed[feature_columns].fillna(0)
        y_train = train_processed['Inhibition']
        X_test = test_processed[feature_columns].fillna(0)
        test_ids = test_processed['ID']
        
        logger.info(f"Features for modeling: {len(feature_columns)}")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Test samples: {len(X_test)}")
        
        # 특성 선택
        selected_features, variance_selector = select_features(X_train, y_train, logger)
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # 스케일링
        logger.info("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # DataFrame으로 변환 (AutoGluon용)
        X_train_final = pd.DataFrame(X_train_scaled, columns=selected_features)
        X_test_final = pd.DataFrame(X_test_scaled, columns=selected_features)
        
        # 타겟 추가
        train_final = X_train_final.copy()
        train_final['target'] = y_train.values
        
        # AutoGluon 모델 학습 (커스텀 메트릭 사용)
        logger.info("Training AutoGluon model with competition metric...")
        predictor = TabularPredictor(
            label='target',
            path=os.path.join(output_dir, 'autogluon_models'),
            eval_metric=competition_scorer  # 커스텀 메트릭 사용
        )
        
        predictor.fit(
            train_data=train_final,
            time_limit=21600,
            presets='best_quality',
            full_weighted_ensemble_additionally=True,
            dynamic_stacking=True,
            fit_strategy='parallel',
            auto_stack=True,
            num_bag_folds=5,
            num_stack_levels=1
        )
        
        # 교차 검증 평가
        logger.info("Performing cross-validation...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        cv_rmse = []
        cv_corr = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_final)):
            logger.info(f"Fold {fold + 1}/5")
            
            X_fold_train = train_final.iloc[train_idx]
            X_fold_val = X_train_final.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            # 임시 predictor 학습 (커스텀 메트릭 사용)
            fold_predictor = TabularPredictor(
                label='target',
                path=os.path.join(output_dir, f'fold_{fold}_models'),
                eval_metric=competition_scorer  # 커스텀 메트릭 사용
            )
            
            fold_predictor.fit(
                train_data=X_fold_train,
                time_limit=300,  # 5분 제한
                presets='medium_quality'
            )
            
            # 검증 세트 예측
            val_preds = fold_predictor.predict(X_fold_val)
            
            # 점수 계산
            score, rmse_norm, corr = calculate_score(y_fold_val, val_preds)
            cv_scores.append(score)
            cv_rmse.append(rmse_norm)
            cv_corr.append(corr)
            
            logger.info(f"Fold {fold + 1} - Score: {score:.4f}, RMSE: {rmse_norm:.4f}, Corr: {corr:.4f}")
        
        logger.info(f"CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        logger.info(f"CV RMSE: {np.mean(cv_rmse):.4f} ± {np.std(cv_rmse):.4f}")
        logger.info(f"CV Correlation: {np.mean(cv_corr):.4f} ± {np.std(cv_corr):.4f}")
        
        # 테스트 예측
        logger.info("Making predictions on test set...")
        test_predictions = predictor.predict(X_test_final)
        
        # 예측값 범위 확인 및 클리핑
        logger.info(f"Test predictions range: {test_predictions.min():.4f} to {test_predictions.max():.4f}")
        test_predictions = np.clip(test_predictions, 0, 100)  # 0-100 범위로 클리핑
        logger.info(f"After clipping: {test_predictions.min():.4f} to {test_predictions.max():.4f}")
        
        # 제출 파일 생성
        submission_df = pd.DataFrame({
            'ID': test_ids,
            'Inhibition': test_predictions
        })
        
        submission_path = os.path.join(output_dir, 'submission.csv')
        submission_df.to_csv(submission_path, index=False, encoding='utf-8')
        logger.info(f"Submission file saved: {submission_path}")
        
        # 모델 성능 리더보드 확인
        logger.info("Model leaderboard:")
        leaderboard = predictor.leaderboard(silent=True)
        logger.info(f"\n{leaderboard}")
        
        # 리더보드를 파일로 저장
        leaderboard.to_csv(os.path.join(output_dir, 'model_leaderboard.csv'), index=False)
        
        # 결과 요약 저장
        summary = {
            'cv_score_mean': float(np.mean(cv_scores)),
            'cv_score_std': float(np.std(cv_scores)),
            'cv_rmse_mean': float(np.mean(cv_rmse)),
            'cv_rmse_std': float(np.std(cv_rmse)),
            'cv_correlation_mean': float(np.mean(cv_corr)),
            'cv_correlation_std': float(np.std(cv_corr)),
            'final_features_count': len(selected_features),
            'train_samples': len(X_train_final),
            'test_samples': len(X_test_final),
            'best_model_score': float(leaderboard['score_val'].max()),
            'best_model': str(leaderboard.loc[leaderboard['score_val'].idxmax(), 'model'])
        }
        
        summary_path = os.path.join(output_dir, 'experiment_summary.json')
        import json
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 전처리된 데이터 저장
        logger.info("Saving processed data...")
        train_final.to_csv(os.path.join(output_dir, 'X_train_processed.csv'), index=False)
        X_test_final.to_csv(os.path.join(output_dir, 'X_test_processed.csv'), index=False)
        test_ids.to_csv(os.path.join(output_dir, 'test_ids.csv'), index=False)
        
        # 전처리 파이프라인 저장
        joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(variance_selector, os.path.join(output_dir, 'variance_selector.pkl'))
        joblib.dump(selected_features, os.path.join(output_dir, 'selected_features.pkl'))
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved in: {output_dir}")
        logger.info(f"Best model: {summary['best_model']} (Score: {summary['best_model_score']:.4f})")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 