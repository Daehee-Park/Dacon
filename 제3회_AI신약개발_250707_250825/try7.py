import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import lightgbm as lgb
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
    'OPTUNA_TRIALS': 30,
    'OUTPUT_DIR': 'output/try7',
    'EXPERIMENT_NAME': 'try7_data_quality_source_aware',
    'RELIABILITY_CV_THRESHOLD': 0.5,
    'IQR_FACTOR': 1.5,
    'SOURCE_CLASSIFIER_THRESHOLD': 0.6
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
        f"{CFG['OUTPUT_DIR']}/models/all_models",
        f"{CFG['OUTPUT_DIR']}/data_quality"
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

def clean_duplicates_within_source(df, source_name, log_file):
    """소스 내 중복 SMILES 정리 및 신뢰도 평가"""
    log_message(f"\n=== {source_name} 중복 정리 시작 ===", log_file)
    
    original_count = len(df)
    log_message(f"원본 데이터: {original_count}개", log_file)
    
    # 중복 SMILES 그룹별 처리
    grouped = df.groupby('smiles')
    
    cleaned_data = []
    reliability_stats = {'High': 0, 'Mid': 0, 'Low': 0}
    
    for smiles, group in grouped:
        ic50_values = group['ic50_nM'].values
        
        if len(ic50_values) == 1:
            # 단일 측정 - High reliability
            cleaned_data.append({
                'smiles': smiles,
                'ic50_nM': ic50_values[0],
                'pIC50': IC50_to_pIC50(ic50_values[0]),
                'source': source_name,
                'reliability': 'High',
                'cv': 0.0,
                'count': 1
            })
            reliability_stats['High'] += 1
            
        else:
            # 다중 측정 - CV 계산
            mean_ic50 = np.mean(ic50_values)
            std_ic50 = np.std(ic50_values)
            cv = std_ic50 / mean_ic50 if mean_ic50 > 0 else float('inf')
            
            if cv <= CFG['RELIABILITY_CV_THRESHOLD']:
                # Mid reliability
                cleaned_data.append({
                    'smiles': smiles,
                    'ic50_nM': mean_ic50,
                    'pIC50': IC50_to_pIC50(mean_ic50),
                    'source': source_name,
                    'reliability': 'Mid',
                    'cv': cv,
                    'count': len(ic50_values)
                })
                reliability_stats['Mid'] += 1
            else:
                # Low reliability - 제거
                reliability_stats['Low'] += 1
    
    cleaned_df = pd.DataFrame(cleaned_data)
    
    log_message(f"정리 결과:", log_file)
    log_message(f"  High reliability: {reliability_stats['High']}개 (단일 측정)", log_file)
    log_message(f"  Mid reliability: {reliability_stats['Mid']}개 (CV ≤ {CFG['RELIABILITY_CV_THRESHOLD']})", log_file)
    log_message(f"  Low reliability (제거): {reliability_stats['Low']}개 (CV > {CFG['RELIABILITY_CV_THRESHOLD']})", log_file)
    log_message(f"  최종 데이터: {len(cleaned_df)}개 ({len(cleaned_df)/original_count*100:.1f}% 유지)", log_file)
    
    return cleaned_df, reliability_stats

def remove_outliers_by_source(df, log_file):
    """소스별 극단값 제거"""
    log_message(f"\n=== 극단값 제거 (IQR {CFG['IQR_FACTOR']}×) ===", log_file)
    
    cleaned_dfs = []
    total_removed = 0
    
    for source in df['source'].unique():
        source_df = df[df['source'] == source].copy()
        original_count = len(source_df)
        
        # IQR 기반 극단값 제거
        Q1 = source_df['pIC50'].quantile(0.25)
        Q3 = source_df['pIC50'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - CFG['IQR_FACTOR'] * IQR
        upper_bound = Q3 + CFG['IQR_FACTOR'] * IQR
        
        source_clean = source_df[(source_df['pIC50'] >= lower_bound) & (source_df['pIC50'] <= upper_bound)]
        removed_count = original_count - len(source_clean)
        total_removed += removed_count
        
        log_message(f"{source}: {removed_count}개 제거 ({removed_count/original_count*100:.1f}%), {len(source_clean)}개 유지", log_file)
        log_message(f"  범위: {lower_bound:.2f} ≤ pIC50 ≤ {upper_bound:.2f}", log_file)
        
        cleaned_dfs.append(source_clean)
    
    final_df = pd.concat(cleaned_dfs, ignore_index=True)
    log_message(f"총 제거: {total_removed}개, 최종: {len(final_df)}개", log_file)
    
    return final_df

def load_and_clean_data():
    """데이터 로딩 및 품질 개선"""
    log_file = f"{CFG['OUTPUT_DIR']}/data_quality/cleaning_log.txt"
    
    log_message("=== 데이터 로딩 및 품질 개선 시작 ===", log_file)
    
    # 원본 데이터 로드
    chembl = pd.read_csv("data/ChEMBL_ASK1(IC50).csv", sep=';')
    pubchem = pd.read_csv("data/Pubchem_ASK1.csv", low_memory=False)
    
    # ChEMBL 전처리
    chembl.columns = chembl.columns.str.strip().str.replace('"', '')
    chembl = chembl[chembl['Standard Type'] == 'IC50']
    chembl = chembl[['Smiles', 'Standard Value']].rename(columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'}).dropna()
    chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')
    chembl = chembl.dropna(subset=['ic50_nM'])
    chembl = chembl[chembl['ic50_nM'] > 0]
    
    # PubChem 전처리
    pubchem = pubchem[['SMILES', 'Activity_Value']].rename(columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'}).dropna()
    pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')
    pubchem = pubchem.dropna(subset=['ic50_nM'])
    pubchem = pubchem[pubchem['ic50_nM'] > 0]
    
    log_message(f"원본 데이터: ChEMBL {len(chembl)}개, PubChem {len(pubchem)}개", log_file)
    
    # 소스별 중복 정리
    chembl_clean, chembl_stats = clean_duplicates_within_source(chembl, 'ChEMBL', log_file)
    pubchem_clean, pubchem_stats = clean_duplicates_within_source(pubchem, 'PubChem', log_file)
    
    # 전체 데이터 합치기
    total_clean = pd.concat([chembl_clean, pubchem_clean], ignore_index=True)
    
    # 극단값 제거
    total_final = remove_outliers_by_source(total_clean, log_file)
    
    # 최종 통계
    log_message(f"\n=== 최종 데이터 품질 요약 ===", log_file)
    log_message(f"전체 데이터: {len(total_final)}개", log_file)
    
    reliability_summary = total_final['reliability'].value_counts()
    for reliability, count in reliability_summary.items():
        log_message(f"  {reliability} reliability: {count}개 ({count/len(total_final)*100:.1f}%)", log_file)
    
    source_summary = total_final['source'].value_counts()
    for source, count in source_summary.items():
        log_message(f"  {source}: {count}개 ({count/len(total_final)*100:.1f}%)", log_file)
    
    # 품질 리포트 저장
    quality_report = {
        'original_data': {'ChEMBL': len(chembl), 'PubChem': len(pubchem)},
        'cleaned_data': {'ChEMBL': len(chembl_clean), 'PubChem': len(pubchem_clean)},
        'final_data': len(total_final),
        'reliability_stats': {
            'ChEMBL': chembl_stats,
            'PubChem': pubchem_stats,
            'final': reliability_summary.to_dict()
        }
    }
    
    with open(f"{CFG['OUTPUT_DIR']}/data_quality/quality_report.json", 'w') as f:
        json.dump(quality_report, f, indent=2)
    
    log_message("=== 데이터 품질 개선 완료 ===", log_file)
    
    return total_final

def create_features(df):
    """피처 생성"""
    log_message("피처 생성 중...")
    
    # Morgan fingerprint
    df['Fingerprint'] = df['smiles'].apply(smiles_to_fingerprint)
    df = df[df['Fingerprint'].notnull()]
    
    # 분자 특성
    mol_props = df['smiles'].apply(calculate_molecular_properties)
    mol_props_df = pd.DataFrame(mol_props.tolist())
    
    # 피처 매트릭스 생성
    X_fp = np.stack(df['Fingerprint'].values)
    
    scaler = StandardScaler()
    X_props = scaler.fit_transform(mol_props_df)
    
    X = np.hstack([X_fp, X_props])
    
    return X, scaler, df.reset_index(drop=True)

def train_source_classifier(df, X, log_file):
    """소스 분류기 학습 (ChEMBL vs PubChem)"""
    log_message("\n=== 소스 분류기 학습 ===", log_file)
    
    # 소스 라벨 생성 (PubChem=1, ChEMBL=0)
    y_source = (df['source'] == 'PubChem').astype(int)
    
    log_message(f"분류 대상: PubChem {sum(y_source)}개, ChEMBL {len(y_source)-sum(y_source)}개", log_file)
    
    # 간단한 LightGBM 분류기
    classifier = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=CFG['SEED'],
        verbose=-1
    )
    
    classifier.fit(X, y_source)
    
    # 성능 확인
    train_proba = classifier.predict_proba(X)[:, 1]  # PubChem 확률
    train_pred = (train_proba > 0.5).astype(int)
    accuracy = np.mean(train_pred == y_source)
    
    log_message(f"분류기 훈련 정확도: {accuracy:.3f}", log_file)
    
    # 분류기 저장
    with open(f"{CFG['OUTPUT_DIR']}/models/source_classifier.pkl", 'wb') as f:
        pickle.dump(classifier, f)
    
    return classifier

def train_source_specific_models(df, X, log_file):
    """소스별 모델 학습"""
    log_message("\n=== 소스별 모델 학습 ===", log_file)
    
    models = {}
    cv_results = {}
    
    for source in ['ChEMBL', 'PubChem']:
        log_message(f"\n--- {source} 모델 학습 ---", log_file)
        
        # 소스별 데이터 분리
        source_mask = df['source'] == source
        X_source = X[source_mask]
        y_source = df[source_mask]['pIC50'].values
        y_ic50_source = df[source_mask]['ic50_nM'].values
        
        log_message(f"{source} 데이터: {len(X_source)}개", log_file)
        
        if len(X_source) < 50:  # 데이터가 너무 적으면 스킵
            log_message(f"{source} 데이터 부족으로 스킵", log_file)
            continue
        
        # CV 설정
        kf = KFold(n_splits=min(CFG['N_FOLDS'], len(X_source)//10), shuffle=True, random_state=CFG['SEED'])
        
        # 모델별 학습
        source_models = {}
        source_cv_results = {}
        
        for model_name in ['CatBoost', 'ExtraTrees']:
            log_message(f"  {model_name} 최적화 중...", log_file)
            
            # 하이퍼파라미터 최적화
            best_params, cv_scores = optimize_source_model(
                model_name, X_source, y_source, y_ic50_source, kf, log_file
            )
            
            # 최종 모델 학습
            if model_name == 'CatBoost':
                model = cb.CatBoostRegressor(**best_params)
            else:  # ExtraTrees
                model = ExtraTreesRegressor(**best_params)
            
            model.fit(X_source, y_source)
            
            source_models[model_name] = model
            source_cv_results[model_name] = {
                'cv_scores': cv_scores,
                'mean_score': np.mean(cv_scores),
                'std_score': np.std(cv_scores),
                'best_params': best_params
            }
            
            log_message(f"  {model_name} CV 점수: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}", log_file)
        
        models[source] = source_models
        cv_results[source] = source_cv_results
        
        # 최적화 결과 저장
        with open(f"{CFG['OUTPUT_DIR']}/model_optimization/{source.lower()}_optimization.json", 'w') as f:
            json.dump(source_cv_results, f, indent=2, default=str)
    
    return models, cv_results

def optimize_source_model(model_name, X, y, y_ic50, kf, log_file):
    """소스별 모델 하이퍼파라미터 최적화"""
    
    def objective(trial):
        try:
            if model_name == 'CatBoost':
                params = {
                    'iterations': trial.suggest_int('iterations', 50, 200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'depth': trial.suggest_int('depth', 3, 6),  # shallow depth
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 5, 20),  # 강한 L2
                    'random_state': CFG['SEED'],
                    'verbose': False,
                    'allow_writing_files': False
                }
                model = cb.CatBoostRegressor(**params)
                
            else:  # ExtraTrees
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 150),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),  # shallow depth
                    'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 3, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                    'random_state': CFG['SEED']
                }
                model = ExtraTreesRegressor(**params)
            
            # CV 평가
            cv_scores = []
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                y_ic50_val = y_ic50[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_ic50_pred = pIC50_to_IC50(y_pred)
                
                normalized_rmse = calculate_normalized_rmse(y_ic50_val, y_ic50_pred)
                correlation_squared = calculate_correlation(y_val, y_pred)
                final_score = calculate_final_score(normalized_rmse, correlation_squared)
                
                cv_scores.append(final_score)
            
            return np.mean(cv_scores)
            
        except Exception as e:
            return 0.0
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=CFG['OPTUNA_TRIALS'])
    
    # 최고 파라미터로 최종 CV 점수 계산
    best_params = study.best_params
    cv_scores = []
    
    if model_name == 'CatBoost':
        model = cb.CatBoostRegressor(**best_params)
    else:
        model = ExtraTreesRegressor(**best_params)
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        y_ic50_val = y_ic50[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_ic50_pred = pIC50_to_IC50(y_pred)
        
        normalized_rmse = calculate_normalized_rmse(y_ic50_val, y_ic50_pred)
        correlation_squared = calculate_correlation(y_val, y_pred)
        final_score = calculate_final_score(normalized_rmse, correlation_squared)
        
        cv_scores.append(final_score)
    
    return best_params, cv_scores

def evaluate_loso_cv(df, X, models, cv_results, log_file):
    """Leave-One-Source-Out CV 평가"""
    log_message("\n=== LOSO CV 평가 ===", log_file)
    
    loso_scores = []
    
    for test_source in ['ChEMBL', 'PubChem']:
        train_source = 'PubChem' if test_source == 'ChEMBL' else 'ChEMBL'
        
        if train_source not in models or test_source not in models:
            log_message(f"{train_source} → {test_source}: 모델 없음으로 스킵", log_file)
            continue
        
        # 데이터 분리
        train_mask = df['source'] == train_source
        test_mask = df['source'] == test_source
        
        X_test = X[test_mask]
        y_test = df[test_mask]['pIC50'].values
        y_ic50_test = df[test_mask]['ic50_nM'].values
        
        # 최고 성능 모델 선택 (CV 결과 기반)
        if train_source in cv_results:
            best_model_name = max(cv_results[train_source].keys(), 
                                key=lambda x: cv_results[train_source][x]['mean_score'])
            model = models[train_source][best_model_name]
            
            log_message(f"{train_source} 최고 모델: {best_model_name} (CV: {cv_results[train_source][best_model_name]['mean_score']:.4f})", log_file)
        else:
            # fallback: 첫 번째 모델 사용
            best_model_name = list(models[train_source].keys())[0]
            model = models[train_source][best_model_name]
            log_message(f"{train_source} 기본 모델: {best_model_name}", log_file)
        
        # 예측
        y_pred = model.predict(X_test)
        y_ic50_pred = pIC50_to_IC50(y_pred)
        
        # 점수 계산
        normalized_rmse = calculate_normalized_rmse(y_ic50_test, y_ic50_pred)
        correlation_squared = calculate_correlation(y_test, y_pred)
        final_score = calculate_final_score(normalized_rmse, correlation_squared)
        
        loso_scores.append(final_score)
        
        log_message(f"{train_source} → {test_source}: {final_score:.4f}", log_file)
    
    if loso_scores:
        mean_loso = np.mean(loso_scores)
        log_message(f"LOSO 평균 점수: {mean_loso:.4f} ± {np.std(loso_scores):.4f}", log_file)
        return mean_loso
    else:
        log_message("LOSO 평가 불가", log_file)
        return 0.0

def make_source_aware_predictions(models, cv_results, source_classifier, X_test, log_file):
    """소스 인식 앙상블 예측"""
    log_message("\n=== 소스 인식 예측 ===", log_file)
    
    # 테스트 데이터의 소스 확률 예측
    source_proba = source_classifier.predict_proba(X_test)[:, 1]  # PubChem 확률
    
    log_message(f"테스트 데이터 소스 분포:", log_file)
    log_message(f"  PubChem 확률 평균: {np.mean(source_proba):.3f}", log_file)
    log_message(f"  확신도 높음 (>0.8 또는 <0.2): {np.sum((source_proba > 0.8) | (source_proba < 0.2))}개", log_file)
    log_message(f"  불확실 (0.2-0.8): {np.sum((source_proba >= 0.2) & (source_proba <= 0.8))}개", log_file)
    
    # 각 소스 모델로 예측
    predictions = {}
    
    for source in models.keys():
        if source in models and source in cv_results:
            # 최고 성능 모델 선택
            best_model_name = max(cv_results[source].keys(), 
                                key=lambda x: cv_results[source][x]['mean_score'])
            model = models[source][best_model_name]
            
            pred = model.predict(X_test)
            predictions[source] = pred
            
            cv_score = cv_results[source][best_model_name]['mean_score']
            log_message(f"{source} 모델({best_model_name}, CV:{cv_score:.4f}) 예측 완료", log_file)
    
    # 소스 인식 앙상블
    if 'ChEMBL' in predictions and 'PubChem' in predictions:
        # 가중 평균
        final_predictions = []
        
        for i, p_pubchem in enumerate(source_proba):
            p_chembl = 1 - p_pubchem
            
            if max(p_pubchem, p_chembl) < CFG['SOURCE_CLASSIFIER_THRESHOLD']:
                # 불확실한 경우 균등 평균
                pred = (predictions['PubChem'][i] + predictions['ChEMBL'][i]) / 2
            else:
                # 가중 평균
                pred = p_pubchem * predictions['PubChem'][i] + p_chembl * predictions['ChEMBL'][i]
            
            final_predictions.append(pred)
        
        final_predictions = np.array(final_predictions)
        
        # 통계
        weighted_count = np.sum([max(p, 1-p) >= CFG['SOURCE_CLASSIFIER_THRESHOLD'] for p in source_proba])
        log_message(f"가중 앙상블: {weighted_count}개, 균등 앙상블: {len(source_proba) - weighted_count}개", log_file)
        
    elif 'PubChem' in predictions:
        log_message("PubChem 모델만 사용", log_file)
        final_predictions = predictions['PubChem']
        
    elif 'ChEMBL' in predictions:
        log_message("ChEMBL 모델만 사용", log_file)
        final_predictions = predictions['ChEMBL']
        
    else:
        raise ValueError("사용 가능한 모델이 없습니다")
    
    return final_predictions, source_proba

def main():
    """메인 실행 함수"""
    
    try:
        # 초기 설정
        seed_everything(CFG['SEED'])
        setup_output_directory()
        
        log_file = f"{CFG['OUTPUT_DIR']}/experiment_log.txt"
        log_message("=== Try7 실험 시작 ===", log_file)
        log_message(f"설정: {CFG}", log_file)
        
        # 1. 데이터 품질 개선
        df_clean = load_and_clean_data()
        
        # 2. 피처 생성
        X, scaler, df_final = create_features(df_clean)
        
        log_message(f"최종 데이터: {len(df_final)}개, 피처: {X.shape[1]}개", log_file)
        
        # 3. 소스 분류기 학습
        source_classifier = train_source_classifier(df_final, X, log_file)
        
        # 4. 소스별 모델 학습
        models, cv_results = train_source_specific_models(df_final, X, log_file)
        
        # 5. LOSO CV 평가
        loso_score = evaluate_loso_cv(df_final, X, models, cv_results, log_file)
        
        # 6. 테스트 데이터 예측
        log_message("\n=== 테스트 데이터 예측 ===", log_file)
        
        # 테스트 데이터 로드
        test_df = pd.read_csv("data/test.csv")
        
        # 테스트 피처 생성
        test_mol_props = test_df['Smiles'].apply(calculate_molecular_properties)
        test_mol_props_df = pd.DataFrame(test_mol_props.tolist())
        
        test_df['Fingerprint'] = test_df['Smiles'].apply(smiles_to_fingerprint)
        test_df = test_df[test_df['Fingerprint'].notnull()]
        
        X_test_fp = np.stack(test_df['Fingerprint'].values)
        X_test_props = scaler.transform(test_mol_props_df.iloc[:len(test_df)])
        X_test = np.hstack([X_test_fp, X_test_props])
        
        # 소스 인식 예측
        final_predictions, source_proba = make_source_aware_predictions(
            models, cv_results, source_classifier, X_test, log_file
        )
        
        # IC50로 변환
        test_ic50_predictions = pIC50_to_IC50(final_predictions)
        
        # 제출 파일 생성
        submission = pd.DataFrame({
            'ID': test_df['ID'],
            'ASK1_IC50_nM': test_ic50_predictions
        })
        
        submission.to_csv(f"{CFG['OUTPUT_DIR']}/submission.csv", index=False)
        
        log_message(f"예측 완료: {len(submission)}개 샘플", log_file)
        
        # 7. 결과 저장
        results_summary = {
            'experiment_name': CFG['EXPERIMENT_NAME'],
            'timestamp': datetime.now().isoformat(),
            'config': CFG,
            'data_quality': {
                'final_data_size': len(df_final),
                'sources': df_final['source'].value_counts().to_dict(),
                'reliability': df_final['reliability'].value_counts().to_dict()
            },
            'cv_results': cv_results,
            'loso_score': loso_score,
            'test_predictions': {
                'count': len(submission),
                'source_distribution': {
                    'pubchem_prob_mean': float(np.mean(source_proba)),
                    'uncertain_samples': int(np.sum((source_proba >= 0.2) & (source_proba <= 0.8)))
                }
            }
        }
        
        with open(f"{CFG['OUTPUT_DIR']}/experiment_summary.json", 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        # 모델 저장
        with open(f"{CFG['OUTPUT_DIR']}/models/trained_models.pkl", 'wb') as f:
            pickle.dump(models, f)
        
        with open(f"{CFG['OUTPUT_DIR']}/models/scaler.pkl", 'wb') as f:
            pickle.dump(scaler, f)
        
        log_message("=== Try7 실험 완료 ===", log_file)
        log_message(f"LOSO CV 점수: {loso_score:.4f}", log_file)
        
        # 성능 요약
        log_message("\n=== 성능 요약 ===", log_file)
        for source in cv_results:
            for model_name in cv_results[source]:
                score = cv_results[source][model_name]['mean_score']
                std = cv_results[source][model_name]['std_score']
                log_message(f"{source} {model_name}: {score:.4f} ± {std:.4f}", log_file)
        
        # 실패 경고
        if loso_score < 0.4:  # 임계값
            warning_msg = f"⚠️ LOSO 점수가 낮습니다 ({loso_score:.4f}). 데이터 품질 재검토 필요."
            log_message(warning_msg, log_file)
            
            with open(f"{CFG['OUTPUT_DIR']}/warning.txt", 'w') as f:
                f.write(warning_msg)
        
    except Exception as e:
        error_msg = f"실험 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        log_message(error_msg, log_file)
        
        with open(f"{CFG['OUTPUT_DIR']}/error_log.txt", 'w', encoding='utf-8') as f:
            f.write(error_msg)
        
        raise

if __name__ == "__main__":
    main() 