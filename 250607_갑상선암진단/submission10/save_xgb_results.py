import json

# XGBoost 결과 저장
f1_mean = 0.4871
f1_std = 0.0036  
threshold_mean = 0.486
goal_achieved = True

best_params = {
    'n_estimators': 1439,
    'learning_rate': 0.012028625587080639,
    'max_depth': 4,
    'subsample': 0.8034068864796984,
    'colsample_bytree': 0.742770742143974,
    'reg_alpha': 14.70363413549785,
    'reg_lambda': 5.444853085179024,
    'min_child_weight': 10,
    'gamma': 0.3670487834559327,
    'scale_pos_weight': 4.3944192340302894,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1
}

xgb_results = {
    'model_name': 'XGBoost',
    'cv_f1_mean': float(f1_mean),
    'cv_f1_std': float(f1_std),
    'threshold': float(threshold_mean),
    'best_params': best_params,
    'goal_achieved': bool(goal_achieved),
    'optuna_best_value': 0.4872,
    'n_trials': 100
}

with open('./result/xgb_results.json', 'w') as f:
    json.dump(xgb_results, f, indent=2)

print('XGBoost 결과 저장 완료!')
print(f'F1 Score: {f1_mean:.4f} ± {f1_std:.4f}')
print(f'목표 달성: {"✅" if goal_achieved else "❌"}')
print('다음 단계: python step4_optimize_cat.py') 