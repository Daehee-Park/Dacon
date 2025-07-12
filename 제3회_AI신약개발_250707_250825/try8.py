import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularPredictor
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from custom_metrics import competition_scorer
import os
import random
import warnings
warnings.filterwarnings('ignore')

CFG = {
    'NBITS': 2048,
    'SEED': 42,
    'TIME_LIMIT': 3600,
    'OUTPUT_DIR': 'output/try8/'
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED'])

# 출력 디렉토리 생성
os.makedirs(CFG['OUTPUT_DIR'], exist_ok=True)

# SMILES 데이터를 분자 지문으로 변환
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=CFG['NBITS'])
        return np.array(fp)
    else:
        return np.zeros((CFG['NBITS'],))

def IC50_to_pIC50(ic50_nM):
    ic50_nM = np.clip(ic50_nM, 1e-10, None)
    return 9 - np.log10(ic50_nM)

def pIC50_to_IC50(pIC50):
    return 10 ** (9 - pIC50)

print("데이터 로딩 및 전처리...")

# ChEMBL 데이터 로딩
chembl = pd.read_csv("data/ChEMBL_ASK1(IC50).csv", sep=';')
chembl.columns = chembl.columns.str.strip().str.replace('"', '')
chembl = chembl[chembl['Standard Type'] == 'IC50']
chembl = chembl[['Smiles', 'Standard Value']].rename(columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'}).dropna()
chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')
chembl['pIC50'] = IC50_to_pIC50(chembl['ic50_nM'])

# PubChem 데이터 로딩
pubchem = pd.read_csv("data/Pubchem_ASK1.csv")
pubchem = pubchem[['SMILES', 'Activity_Value']].rename(columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'}).dropna()
pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')
pubchem['pIC50'] = IC50_to_pIC50(pubchem['ic50_nM'])

# 데이터 통합
total = pd.concat([chembl, pubchem], ignore_index=True)
total = total.drop_duplicates(subset='smiles')
total = total[total['ic50_nM'] > 0].dropna()

print(f"총 훈련 데이터: {len(total)}개")

# 분자 지문 생성
total['Fingerprint'] = total['smiles'].apply(smiles_to_fingerprint)
total = total[total['Fingerprint'].notnull()]

# 지문을 개별 열로 변환
X_fp = np.stack(total['Fingerprint'].values)
fp_df = pd.DataFrame(X_fp, columns=[f'fp_{i}' for i in range(CFG['NBITS'])])

# 최종 훈련 데이터 구성
train_data = pd.concat([
    total[['smiles']].reset_index(drop=True),
    fp_df,
    total[['pIC50']].reset_index(drop=True)
], axis=1)

print(f"최종 훈련 데이터 형태: {train_data.shape}")
print(f"Target 범위: {train_data['pIC50'].min():.3f} ~ {train_data['pIC50'].max():.3f}")

# AutoGluon 모델 학습
print("\nAutoGluon 모델 학습 시작...")

# Feature Generator 설정
feature_generator = AutoMLPipelineFeatureGenerator(
    enable_numeric_features=True,
    enable_categorical_features=True,
    enable_datetime_features=False,
    enable_text_special_features=False,
    enable_text_ngram_features=False,
    enable_raw_text_features=False,
    enable_vision_features=False
)

# TabularPredictor 초기화 (feature_generator 제거)
predictor = TabularPredictor(
    label='pIC50',
    path=CFG['OUTPUT_DIR'] + 'models',
    eval_metric=competition_scorer
)

# 모델 학습 (feature_generator를 fit 메서드에 전달)
predictor.fit(
    train_data,
    time_limit=CFG['TIME_LIMIT'],
    presets='best_quality',
    holdout_frac=0.2,
    feature_generator=feature_generator,
    ag_args_ensemble={
        'full_weighted_ensemble_additionally': True,
        'dynamic_stacking': True,
        'auto_stack': True,
        'num_bag_folds': 5,
        'num_stack_levels': 1
    }
)

print("\n모델 학습 완료!")

# 모델 성능 평가
print("\n=== 모델 성능 평가 ===")
leaderboard = predictor.leaderboard(silent=True)
print(leaderboard)

# 리더보드 저장
leaderboard.to_csv(CFG['OUTPUT_DIR'] + 'model_leaderboard.csv', index=False)

# 테스트 데이터 예측
print("\n테스트 데이터 예측...")
test = pd.read_csv("data/test.csv")
test['Fingerprint'] = test['Smiles'].apply(smiles_to_fingerprint)
test = test[test['Fingerprint'].notnull()]

# 테스트 데이터 지문을 개별 열로 변환
X_test_fp = np.stack(test['Fingerprint'].values)
test_fp_df = pd.DataFrame(X_test_fp, columns=[f'fp_{i}' for i in range(CFG['NBITS'])])

# 테스트 데이터 구성
test_data = pd.concat([
    test[['Smiles']].reset_index(drop=True).rename(columns={'Smiles': 'smiles'}),
    test_fp_df
], axis=1)

# 예측 수행
test_pred = predictor.predict(test_data)
test['pIC50_pred'] = test_pred
test['ASK1_IC50_nM'] = pIC50_to_IC50(test['pIC50_pred'])

# 제출 파일 생성
submission = pd.read_csv('data/sample_submission.csv')
submission['ASK1_IC50_nM'] = test['ASK1_IC50_nM']
submission.to_csv(CFG['OUTPUT_DIR'] + 'submission.csv', index=False)

print(f"\n예측 완료! 제출 파일 저장: {CFG['OUTPUT_DIR']}submission.csv")
print(f"예측값 범위: {test['ASK1_IC50_nM'].min():.3f} ~ {test['ASK1_IC50_nM'].max():.3f}")

# 실험 요약 저장
summary = {
    'experiment': 'Try8_AutoGluon_Baseline',
    'train_samples': len(total),
    'test_samples': len(test),
    'features': ['MorganFP_2048', 'AutoMLPipelineFeatureGenerator'],
    'model': 'AutoGluon_TabularPredictor',
    'time_limit': CFG['TIME_LIMIT'],
    'best_model': predictor.get_model_best(),
    'validation_score': predictor.get_model_best_score(),
    'config': {
        'presets': 'best_quality',
        'holdout_frac': 0.2,
        'feature_generator': 'AutoMLPipelineFeatureGenerator',
        'hyperparameters': 'multi_model',
        'ag_args_ensemble': {
            'full_weighted_ensemble_additionally': True,
            'dynamic_stacking': True,
            'auto_stack': True,
            'num_bag_folds': 5,
            'num_stack_levels': 1
        }
    }
}

import json
with open(CFG['OUTPUT_DIR'] + 'experiment_summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n실험 요약 저장: {CFG['OUTPUT_DIR']}experiment_summary.json") 