import pandas as pd
import numpy as np
import json
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

RESULT_PATH = './result/'

print("=== Step 10: 최종 예측 및 제출 파일 생성 ===")

# 최종 요약 로드
with open(RESULT_PATH + 'final_summary.json', 'r') as f:
    summary = json.load(f)

best_model_name = summary['best_model']
print(f"최고 성능 모델: {best_model_name}")
print(f"CV F1 Score: {summary['best_f1_score']:.4f} ± {summary['best_f1_std']:.4f}")

# 전처리된 데이터 로드
X_train = pd.read_csv(RESULT_PATH + 'X_train.csv')
X_test = pd.read_csv(RESULT_PATH + 'X_test.csv')
y_train = pd.read_csv(RESULT_PATH + 'y_train.csv')['Cancer']
test_ids = pd.read_csv(RESULT_PATH + 'test_ids.csv')['ID']

print(f"데이터 로드 완료")
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# 최고 모델 파라미터 로드
if best_model_name == 'LightGBM':
    with open(RESULT_PATH + 'lgb_results.json', 'r') as f:
        result = json.load(f)
    
    best_params = result['best_params']
    threshold = result['threshold']
    
    print(f"LightGBM 파라미터 로드 완료")
    print(f"최적 threshold: {threshold:.3f}")
    
    # 모델 학습
    print(f"\n모델 학습 중...")
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)
    
    # 테스트 예측
    print(f"테스트 예측 중...")
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

else:
    print(f"❌ {best_model_name} 모델은 아직 구현되지 않았습니다.")
    print("LightGBM으로 대체하여 예측합니다.")
    
    # LightGBM으로 대체
    with open(RESULT_PATH + 'lgb_results.json', 'r') as f:
        result = json.load(f)
    
    best_params = result['best_params']
    threshold = result['threshold']
    
    final_model = lgb.LGBMClassifier(**best_params)
    final_model.fit(X_train, y_train)
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= threshold).astype(int)

# 예측 결과 분석
pred_positive_rate = test_pred.mean()
pred_distribution = pd.Series(test_pred).value_counts().to_dict()

print(f"\n=== 예측 결과 ===")
print(f"예측 양성률: {pred_positive_rate:.4f} ({test_pred.sum()}/{len(test_pred)})")
print(f"예측 분포: {pred_distribution}")

# 제출 파일 생성
submission = pd.DataFrame({
    'ID': test_ids,
    'Cancer': test_pred
})

submission.to_csv('submission10.csv', index=False)
print(f"\n제출 파일 생성 완료: submission10.csv")

# 최종 결과 요약
final_result = {
    'best_model': best_model_name,
    'cv_f1_score': summary['best_f1_score'],
    'threshold': float(threshold),
    'predicted_positive_rate': float(pred_positive_rate),
    'prediction_distribution': pred_distribution,
    'submission_file': 'submission10.csv'
}

with open(RESULT_PATH + 'final_result.json', 'w') as f:
    json.dump(final_result, f, indent=2)

print(f"최종 결과 저장: final_result.json")

print(f"\n=== 10차 시도 완료 ===")
print(f"✅ 최고 성능: {best_model_name} - CV F1: {summary['best_f1_score']:.4f}")
print(f"✅ 목표 달성: {'✅' if summary['best_f1_score'] > 0.487 else '❌'} (목표 F1 > 0.487)")
print(f"✅ 제출 파일: submission10.csv") 