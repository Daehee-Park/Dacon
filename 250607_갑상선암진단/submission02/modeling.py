import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
import optuna
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = '../data/'
RESULT_PATH = './result/'

# 결과 폴더 확인
import os
if not os.path.exists(RESULT_PATH):
    print("❌ Result folder not found. Run preprocess.py first!")
    exit(1)

print("=== Advanced Modeling: Reducing False Negatives ===")

# 데이터 로드
X_train = pd.read_csv(RESULT_PATH + 'X_train_enhanced.csv')
y_train = pd.read_csv(RESULT_PATH + 'y_train_enhanced.csv').squeeze()
X_test = pd.read_csv(RESULT_PATH + 'X_test_enhanced.csv')

print(f"Enhanced train shape: {X_train.shape}")
print(f"Enhanced test shape: {X_test.shape}")

# 클래스 불균형 정보
pos_ratio = y_train.mean()
neg_ratio = 1 - pos_ratio
print(f"Positive ratio: {pos_ratio:.4f}")

print("\n--- 1. Cost-Sensitive Optimization ---")

# False Negative에 더 큰 페널티 부여
FN_COST = 3.0  # False Negative cost (암을 놓치는 것이 더 위험)
FP_COST = 1.0  # False Positive cost

def custom_f1_score(y_true, y_pred, fn_weight=2.0):
    """False Negative에 더 큰 가중치를 주는 custom F1"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Weighted recall (FN에 더 큰 페널티)
    weighted_recall = tp / (tp + fn_weight * fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # F1 with weighted recall
    if (weighted_recall + precision) > 0:
        f1 = 2 * (precision * weighted_recall) / (precision + weighted_recall)
    else:
        f1 = 0
    
    return f1

def objective_cost_sensitive(trial):
    """Cost-sensitive hyperparameter optimization"""
    
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 8.0, 15.0),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1
    }
    
    # 3-fold CV
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Sample weights로 cost-sensitive learning
        sample_weights = np.ones(len(y_tr))
        sample_weights[y_tr == 1] = FN_COST * neg_ratio / pos_ratio
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, 
                  sample_weight=sample_weights,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
        
        # Lower threshold for higher recall
        val_proba = model.predict_proba(X_val)[:, 1]
        
        best_score = 0
        for threshold in np.arange(0.1, 0.5, 0.02):
            val_preds = (val_proba >= threshold).astype(int)
            
            # Custom F1 with FN penalty
            score = custom_f1_score(y_val, val_preds, fn_weight=FN_COST)
            best_score = max(best_score, score)
        
        scores.append(best_score)
    
    return np.mean(scores)

# Optuna optimization
study = optuna.create_study(direction='maximize', study_name='cost_sensitive')
study.optimize(objective_cost_sensitive, n_trials=25, show_progress_bar=True)

print(f"Best score: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

print("\n--- 2. Building Diverse Base Models ---")

# 최적 파라미터로 base models 설정
best_lgb_params = study.best_params.copy()
best_lgb_params.update({
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1
})

# 다양한 모델 설정
base_models = {
    'lgb_cost': lgb.LGBMClassifier(**best_lgb_params),
    
    'lgb_dart': lgb.LGBMClassifier(
        objective='binary',
        boosting_type='dart',
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=50,
        scale_pos_weight=10,
        drop_rate=0.1,
        skip_drop=0.5,
        seed=42,
        n_jobs=-1,
        verbose=-1
    ),
    
    'xgb_cost': xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=10,
        gamma=1.0,  # 더 conservative한 split
        min_child_weight=5,
        random_state=42,
        n_jobs=-1
    ),
    
    'cat_cost': CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        border_count=128,
        random_seed=42,
        verbose=False,
        eval_metric='F1',
        class_weights={0: 1, 1: FN_COST}
    ),
    
    'rf_balanced': RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight={0: 1, 1: FN_COST * 2},  # 더 aggressive
        random_state=42,
        n_jobs=-1
    ),
    
    'et_balanced': ExtraTreesClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight={0: 1, 1: FN_COST * 2},
        random_state=42,
        n_jobs=-1
    )
}

print("\n--- 3. Training Base Models with Cost-Sensitive Learning ---")

# Cross-validation
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# 결과 저장
base_predictions = {}
test_predictions = {}
model_scores = {}

for model_name, model in base_models.items():
    print(f"\nTraining {model_name}...")
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Sample weights for cost-sensitive learning
        sample_weights = np.ones(len(y_tr))
        sample_weights[y_tr == 1] = FN_COST * neg_ratio / pos_ratio
        
        # Model-specific training
        if model_name.startswith('lgb'):
            model.fit(X_tr, y_tr,
                      sample_weight=sample_weights,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        elif model_name.startswith('xgb'):
            model.fit(X_tr, y_tr, sample_weight=sample_weights)
        elif model_name.startswith('cat'):
            model.fit(X_tr, y_tr,
                      eval_set=(X_val, y_val),
                      early_stopping_rounds=50,
                      verbose=False)
        else:  # RF, ET
            model.fit(X_tr, y_tr, sample_weight=sample_weights)
        
        # Predictions
        val_proba = model.predict_proba(X_val)[:, 1]
        
        # Find optimal threshold for this fold
        best_f1 = 0
        best_thresh = 0.5
        for threshold in np.arange(0.15, 0.6, 0.01):
            val_preds = (val_proba >= threshold).astype(int)
            f1 = f1_score(y_val, val_preds)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = threshold
        
        # Calculate metrics
        val_preds = (val_proba >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, val_preds).ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        fold_scores.append({
            'f1': best_f1,
            'threshold': best_thresh,
            'recall': recall,
            'fn_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        })
        
        oof_preds[val_idx] = val_proba
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    
    # Save results
    base_predictions[model_name] = oof_preds
    test_predictions[model_name] = test_preds
    model_scores[model_name] = fold_scores
    
    # Print average performance
    avg_f1 = np.mean([s['f1'] for s in fold_scores])
    avg_recall = np.mean([s['recall'] for s in fold_scores])
    avg_fn_rate = np.mean([s['fn_rate'] for s in fold_scores])
    
    print(f"{model_name}: F1={avg_f1:.4f}, Recall={avg_recall:.4f}, FN_rate={avg_fn_rate:.4f}")

print("\n--- 4. Stacking Ensemble ---")

# Prepare meta features
meta_train = pd.DataFrame(base_predictions)
meta_test = pd.DataFrame(test_predictions)

# Meta model (LogisticRegression for interpretability)
meta_model = LogisticRegression(
    C=0.5,
    class_weight={0: 1, 1: FN_COST},
    random_state=42,
    max_iter=1000
)

# Train meta model with CV
meta_oof = np.zeros(len(X_train))
meta_test_preds = np.zeros(len(X_test))

for train_idx, val_idx in skf.split(meta_train, y_train):
    meta_model.fit(meta_train.iloc[train_idx], y_train.iloc[train_idx])
    meta_oof[val_idx] = meta_model.predict_proba(meta_train.iloc[val_idx])[:, 1]
    meta_test_preds += meta_model.predict_proba(meta_test)[:, 1] / N_SPLITS

print("\n--- 5. Final Threshold Optimization ---")

# Optimize threshold to minimize false negatives
best_f1 = 0
best_threshold = 0.5
best_metrics = {}

for threshold in np.arange(0.1, 0.5, 0.005):
    preds = (meta_oof >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_train, preds).ravel()
    
    f1 = f1_score(y_train, preds)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Prioritize low FN rate while maintaining reasonable F1
    if f1 > best_f1 and fn_rate < 0.4:  # Max 40% false negative rate
        best_f1 = f1
        best_threshold = threshold
        best_metrics = {
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'fn_rate': fn_rate,
            'fn_count': fn,
            'tp_count': tp
        }

print(f"\nOptimal threshold: {best_threshold:.3f}")
print(f"F1: {best_metrics['f1']:.4f}")
print(f"Recall: {best_metrics['recall']:.4f} (catching {best_metrics['recall']*100:.1f}% of cancer cases)")
print(f"Precision: {best_metrics['precision']:.4f}")
print(f"False Negative Rate: {best_metrics['fn_rate']:.4f} (missing {best_metrics['fn_rate']*100:.1f}% of cancer cases)")
print(f"False Negatives: {best_metrics['fn_count']} out of {best_metrics['fn_count'] + best_metrics['tp_count']} cancer cases")

# Final predictions
final_preds = (meta_oof >= best_threshold).astype(int)
print("\nFinal Classification Report:")
print(classification_report(y_train, final_preds))

print("\n--- 6. Saving Enhanced Predictions ---")

# Save predictions
np.save(RESULT_PATH + 'stacking_test_preds.npy', meta_test_preds)
np.save(RESULT_PATH + 'stacking_oof_preds.npy', meta_oof)

# Save optimal threshold
import pickle
with open(RESULT_PATH + 'stacking_threshold.pkl', 'wb') as f:
    pickle.dump(best_threshold, f)

# Save model scores for analysis
with open(RESULT_PATH + 'model_scores.pkl', 'wb') as f:
    pickle.dump(model_scores, f)

print("\n✅ Enhanced modeling completed!")
print(f"Expected improvement: Reduced FN rate from 54% to ~{best_metrics['fn_rate']*100:.0f}%")
