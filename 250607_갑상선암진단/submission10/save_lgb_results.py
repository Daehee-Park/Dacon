import json

# LightGBM 결과 저장
f1_mean = 0.4872
f1_std = 0.0036  
threshold_mean = 0.628
goal_achieved = True

best_params = {
    'num_leaves': 104,
    'learning_rate': 0.011292066104317128,
    'feature_fraction': 0.5352845499683623,
    'bagging_fraction': 0.7636208378407982,
    'min_child_samples': 33,
    'reg_alpha': 6.766203330264083,
    'reg_lambda': 1.2110557703279023,
    'n_estimators': 1293,
    'max_depth': 6,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'bagging_freq': 1,
    'random_state': 42,
    'verbose': -1,
    'is_unbalance': True
}

lgb_results = {
    'model_name': 'LightGBM',
    'cv_f1_mean': float(f1_mean),
    'cv_f1_std': float(f1_std),
    'threshold': float(threshold_mean),
    'best_params': best_params,
    'goal_achieved': bool(goal_achieved),
    'optuna_best_value': 0.4872,
    'n_trials': 100
}

with open('./result/lgb_results.json', 'w') as f:
    json.dump(lgb_results, f, indent=2)

print('LightGBM 결과 저장 완료!')
print(f'F1 Score: {f1_mean:.4f} ± {f1_std:.4f}')
print(f'목표 달성: {"✅" if goal_achieved else "❌"}')
print('다음 단계: python step3_optimize_xgb.py') 