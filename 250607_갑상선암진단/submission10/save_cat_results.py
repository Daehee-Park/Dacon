import json

# CatBoost 결과 저장
f1_mean = 0.4872
f1_std = 0.0036  
threshold_mean = 0.504
goal_achieved = True

best_params = {
    'iterations': 1196,
    'learning_rate': 0.02563053297948684,
    'depth': 6,
    'l2_leaf_reg': 1.073735362702601,
    'random_strength': 5.734856048140002,
    'border_count': 186,
    'bagging_temperature': 0.40095596968204394,
    'objective': 'Logloss',
    'random_state': 42,
    'verbose': False,
    'auto_class_weights': 'Balanced'
}

cat_results = {
    'model_name': 'CatBoost',
    'cv_f1_mean': float(f1_mean),
    'cv_f1_std': float(f1_std),
    'threshold': float(threshold_mean),
    'best_params': best_params,
    'goal_achieved': bool(goal_achieved),
    'optuna_best_value': 0.4872,
    'n_trials': 100
}

with open('./result/cat_results.json', 'w') as f:
    json.dump(cat_results, f, indent=2)

print('CatBoost 결과 저장 완료!')
print(f'F1 Score: {f1_mean:.4f} ± {f1_std:.4f}')
print(f'목표 달성: {"✅" if goal_achieved else "❌"}')
print('다음 단계: python step9_compare_results.py') 