import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Hyperparameter Optimization
import optuna
from optuna.samplers import TPESampler

# Utils
import pickle
import os

print("=== 5th Attempt: Binary F1 Score Optimization ===")

# Paths
RESULT_PATH = './result/'
os.makedirs(RESULT_PATH, exist_ok=True)

# Load enhanced data
print("Loading enhanced features...")
X_train = pd.read_csv(RESULT_PATH + 'X_train_enhanced.csv')
y_train = pd.read_csv(RESULT_PATH + 'y_train_enhanced.csv')['Cancer']
X_test = pd.read_csv(RESULT_PATH + 'X_test_enhanced.csv')
test_ids = pd.read_csv(RESULT_PATH + 'test_ids_enhanced.csv')

print(f"Enhanced train shape: {X_train.shape}")
print(f"Enhanced test shape: {X_test.shape}")

# Class distribution
pos_ratio = y_train.mean()
neg_ratio = 1 - pos_ratio
print(f"Positive ratio: {pos_ratio:.4f}")

def f1_objective(y_true, y_pred_proba, threshold=0.5):
    """F1 score for threshold optimization"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return f1_score(y_true, y_pred)

def objective_f1_lgb(trial):
    """F1-optimized LightGBM hyperparameter tuning"""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'scale_pos_weight': neg_ratio / pos_ratio,  # Handle imbalance
        'verbosity': -1,
        'random_state': 42
    }
    
    # Cross-validation for F1 optimization
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    f1_scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Train model
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # Predict and find optimal threshold
        val_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Find best threshold for F1
        best_f1 = 0
        for threshold in np.arange(0.1, 0.8, 0.02):
            f1 = f1_objective(y_val, val_pred_proba, threshold)
            if f1 > best_f1:
                best_f1 = f1
        
        f1_scores.append(best_f1)
    
    return np.mean(f1_scores)

print("\n--- 1. F1-Optimized Hyperparameter Tuning ---")

# Optimize LightGBM for F1
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    study_name='f1_optimization'
)

study.optimize(objective_f1_lgb, n_trials=30, show_progress_bar=True)

print(f"Best F1 score: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

best_lgb_params = study.best_params.copy()
best_lgb_params.update({
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'scale_pos_weight': neg_ratio / pos_ratio,
    'verbosity': -1,
    'random_state': 42
})

print("\n--- 2. Building F1-Optimized Ensemble ---")

# Define base models optimized for F1
base_models = {
    'lgb_f1': lgb.LGBMClassifier(
        **best_lgb_params,
        n_estimators=1000,
        early_stopping_rounds=50
    ),
    
    'lgb_balanced': lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        num_leaves=50,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        max_depth=7,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=neg_ratio / pos_ratio,
        n_estimators=1000,
        random_state=42,
        verbosity=-1
    ),
    
    'xgb_f1': xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        scale_pos_weight=neg_ratio / pos_ratio,
        n_estimators=1000,
        random_state=42,
        verbosity=0
    ),
    
    'rf_f1': RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    
    'et_f1': ExtraTreesClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
}

print("\n--- 3. Training Base Models with F1 Optimization ---")

# Cross-validation setup
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Store results
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
        
        # Model-specific training
        if model_name.startswith('lgb'):
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
        elif model_name.startswith('xgb'):
            model.fit(X_tr, y_tr)
        else:  # RF, ET
            model.fit(X_tr, y_tr)
        
        # Predictions
        val_proba = model.predict_proba(X_val)[:, 1]
        
        # Find optimal threshold for F1
        best_f1 = 0
        best_thresh = 0.5
        for threshold in np.arange(0.1, 0.8, 0.01):
            f1 = f1_objective(y_val, val_proba, threshold)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = threshold
        
        # Calculate metrics
        val_preds = (val_proba >= best_thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, val_preds).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        fold_scores.append({
            'f1': best_f1,
            'threshold': best_thresh,
            'precision': precision,
            'recall': recall
        })
        
        oof_preds[val_idx] = val_proba
        test_preds += model.predict_proba(X_test)[:, 1] / N_SPLITS
    
    # Save results
    base_predictions[model_name] = oof_preds
    test_predictions[model_name] = test_preds
    model_scores[model_name] = fold_scores
    
    # Print average performance
    avg_f1 = np.mean([s['f1'] for s in fold_scores])
    avg_precision = np.mean([s['precision'] for s in fold_scores])
    avg_recall = np.mean([s['recall'] for s in fold_scores])
    
    print(f"{model_name}: F1={avg_f1:.4f}, Precision={avg_precision:.4f}, Recall={avg_recall:.4f}")

print("\n--- 4. Advanced Ensemble Strategies ---")

# Strategy 1: Simple Average
simple_ensemble = np.mean([base_predictions[name] for name in base_models.keys()], axis=0)

# Strategy 2: Weighted Average (based on CV F1 scores)
weights = {}
for model_name in base_models.keys():
    avg_f1 = np.mean([s['f1'] for s in model_scores[model_name]])
    weights[model_name] = avg_f1

total_weight = sum(weights.values())
weights = {k: v/total_weight for k, v in weights.items()}

weighted_ensemble = np.zeros(len(X_train))
for model_name, weight in weights.items():
    weighted_ensemble += base_predictions[model_name] * weight

# Strategy 3: Stacking with LogisticRegression
meta_train = pd.DataFrame(base_predictions)
meta_test = pd.DataFrame(test_predictions)

meta_model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    random_state=42,
    max_iter=1000
)

# Cross-validation for meta model
stacking_oof = np.zeros(len(X_train))
stacking_test = np.zeros(len(X_test))

for train_idx, val_idx in skf.split(meta_train, y_train):
    meta_model.fit(meta_train.iloc[train_idx], y_train.iloc[train_idx])
    stacking_oof[val_idx] = meta_model.predict_proba(meta_train.iloc[val_idx])[:, 1]
    stacking_test += meta_model.predict_proba(meta_test)[:, 1] / N_SPLITS

print("\n--- 5. Ensemble Comparison and Selection ---")

ensembles = {
    'simple_avg': simple_ensemble,
    'weighted_avg': weighted_ensemble,
    'stacking': stacking_oof
}

best_ensemble = None
best_f1 = 0
best_threshold = 0.5
best_name = ''

for ensemble_name, ensemble_preds in ensembles.items():
    # Find optimal threshold
    ensemble_best_f1 = 0
    ensemble_best_thresh = 0.5
    
    for threshold in np.arange(0.1, 0.8, 0.005):
        f1 = f1_objective(y_train, ensemble_preds, threshold)
        if f1 > ensemble_best_f1:
            ensemble_best_f1 = f1
            ensemble_best_thresh = threshold
    
    print(f"{ensemble_name}: F1={ensemble_best_f1:.4f}, Threshold={ensemble_best_thresh:.3f}")
    
    if ensemble_best_f1 > best_f1:
        best_f1 = ensemble_best_f1
        best_threshold = ensemble_best_thresh
        best_ensemble = ensemble_preds
        best_name = ensemble_name

print(f"\n🏆 Best ensemble: {best_name}")
print(f"Best F1: {best_f1:.4f}")
print(f"Best threshold: {best_threshold:.3f}")

# Final predictions with best ensemble
final_preds = (best_ensemble >= best_threshold).astype(int)

# Calculate final metrics
tn, fp, fn, tp = confusion_matrix(y_train, final_preds).ravel()
final_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
final_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nFinal Performance:")
print(f"F1: {best_f1:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"Recall: {final_recall:.4f}")
print(f"Predicted positive rate: {final_preds.mean():.4f}")

print("\nFinal Classification Report:")
print(classification_report(y_train, final_preds))

print("\n--- 6. Saving F1-Optimized Results ---")

# Select test predictions based on best ensemble
if best_name == 'simple_avg':
    final_test_preds = np.mean([test_predictions[name] for name in base_models.keys()], axis=0)
elif best_name == 'weighted_avg':
    final_test_preds = np.zeros(len(X_test))
    for model_name, weight in weights.items():
        final_test_preds += test_predictions[model_name] * weight
else:  # stacking
    final_test_preds = stacking_test

# Save predictions
np.save(RESULT_PATH + 'f1_optimized_test_preds.npy', final_test_preds)
np.save(RESULT_PATH + 'f1_optimized_oof_preds.npy', best_ensemble)

# Save optimal threshold
with open(RESULT_PATH + 'f1_optimized_threshold.pkl', 'wb') as f:
    pickle.dump(best_threshold, f)

# Save model info
model_info = {
    'best_ensemble': best_name,
    'best_f1': best_f1,
    'best_threshold': best_threshold,
    'weights': weights if best_name == 'weighted_avg' else None,
    'model_scores': model_scores
}

with open(RESULT_PATH + 'f1_model_info.pkl', 'wb') as f:
    pickle.dump(model_info, f)

print("\n✅ F1-optimized modeling completed!")
print(f"Target: Beat current best (Public F1: 0.5109)")
print(f"Expected: F1 > 0.52 with enhanced features + F1 optimization") 