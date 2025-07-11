import json
import pandas as pd
import os

RESULT_PATH = './result/'

print("=== Step 9: 모델 성능 비교 및 최고 모델 선택 ===")

# 모든 모델 결과 로드
model_files = {
    'LightGBM': 'lgb_results.json',
    'XGBoost': 'xgb_results.json', 
    'CatBoost': 'cat_results.json',
    'ExtraTrees': 'et_results.json',
    'LogisticRegression': 'lr_results.json',
    'MLP': 'mlp_results.json',
    'TabNet': 'tabnet_results.json'
}

all_results = {}

for model_name, filename in model_files.items():
    filepath = RESULT_PATH + filename
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            result = json.load(f)
            all_results[model_name] = result
        print(f"✅ {model_name} 결과 로드: F1 = {result['cv_f1_mean']:.4f}")
    else:
        print(f"❌ {model_name} 결과 없음 ({filename})")

print(f"\n로드된 모델 수: {len(all_results)}")

if len(all_results) == 0:
    print("❌ 로드된 모델 결과가 없습니다.")
    exit()

# 성능 순위 매기기
sorted_results = sorted(all_results.items(), key=lambda x: x[1]['cv_f1_mean'], reverse=True)

print(f"\n=== 모델 성능 순위 ===")
for i, (model_name, result) in enumerate(sorted_results, 1):
    f1_score = result['cv_f1_mean']
    f1_std = result['cv_f1_std']
    threshold = result['threshold']
    goal_check = "✅" if result['goal_achieved'] else "❌"
    
    print(f"{i}. {model_name}: F1 = {f1_score:.4f} ± {f1_std:.4f} (Threshold: {threshold:.3f}) {goal_check}")

# 최고 성능 모델 선택
best_model_name, best_result = sorted_results[0]
print(f"\n=== 최고 성능 모델 ===")
print(f"모델: {best_model_name}")
print(f"CV F1 Score: {best_result['cv_f1_mean']:.4f} ± {best_result['cv_f1_std']:.4f}")
print(f"Threshold: {best_result['threshold']:.3f}")
print(f"목표 달성: {'✅' if best_result['goal_achieved'] else '❌'}")

# 목표 달성 모델 수 계산
goal_achieved_count = sum(1 for _, result in all_results.items() if result['goal_achieved'])
print(f"\n목표 달성 모델 수: {goal_achieved_count}/{len(all_results)}")

# 최종 요약 저장
summary = {
    'best_model': best_model_name,
    'best_f1_score': best_result['cv_f1_mean'],
    'best_f1_std': best_result['cv_f1_std'],
    'best_threshold': best_result['threshold'],
    'goal_achieved_count': goal_achieved_count,
    'total_models': len(all_results),
    'all_results': {name: {
        'f1_mean': result['cv_f1_mean'],
        'f1_std': result['cv_f1_std'],
        'threshold': result['threshold'],
        'goal_achieved': result['goal_achieved']
    } for name, result in all_results.items()}
}

with open(RESULT_PATH + 'final_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n최종 요약 저장: final_summary.json")
print(f"\n=== Step 9 완료 ===")
print(f"다음 단계: python step10_final_prediction.py") 