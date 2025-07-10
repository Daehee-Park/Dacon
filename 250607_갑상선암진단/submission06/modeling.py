import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, BayesianRidge, ElasticNet, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

# Advanced Optimization
import optuna
from optuna.samplers import TPESampler
from scipy.optimize import minimize
from sklearn.model_selection import cross_val_score

# Alternative to Neural Networks
from sklearn.neural_network import MLPClassifier

# Utils
import pickle
import os
from datetime import datetime
import time

print("=== 6th Attempt: Advanced Ensemble & Deep Optimization ===")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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

# Set up reproducibility
SEED = 42
np.random.seed(SEED)

def f1_objective(y_true, y_pred_proba, threshold=0.5):
    """F1 score for threshold optimization"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    return f1_score(y_true, y_pred)

def create_mlp_classifier():
    """Create advanced MLPClassifier for meta-learning"""
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.01,
        batch_size=512,
        learning_rate_init=0.001,
        max_iter=2000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=SEED
    )

def objective_advanced_lgb(trial):
    """Advanced LightGBM hyperparameter tuning with realistic ranges"""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt']),  # Focus on GBDT
        'num_leaves': trial.suggest_int('num_leaves', 30, 120),  # Narrowed range
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15),  # Realistic range
        'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),  # Higher values
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.7, 1.0),  # Higher values
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),  # Smaller range
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),  # Realistic range
        'max_depth': trial.suggest_int('max_depth', 5, 10),  # Focused range
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),  # Narrowed
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),  # Narrowed
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 5.0, 15.0),  # Focused range
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.5),  # Smaller range
        'verbosity': -1,
        'random_state': SEED
    }
    
    # No DART parameters needed since focusing on GBDT
    
    # Cross-validation for robust evaluation
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    f1_scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Train model with advanced settings
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        # Predict and optimize threshold
        val_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
        
        # Find best threshold for F1
        best_f1 = 0
        for threshold in np.arange(0.05, 0.9, 0.025):
            f1 = f1_objective(y_val, val_pred_proba, threshold)
            if f1 > best_f1:
                best_f1 = f1
        
        f1_scores.append(best_f1)
    
    return np.mean(f1_scores)

print("\n--- 1. Advanced Hyperparameter Optimization (50 trials) ---")

# Optimize LightGBM with advanced settings
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=SEED, n_startup_trials=20),
    study_name='advanced_optimization',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
)

study.optimize(objective_advanced_lgb, n_trials=50, show_progress_bar=True)

print(f"Best F1 score: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

best_lgb_params = study.best_params.copy()
best_lgb_params.update({
    'objective': 'binary',
    'metric': 'binary_logloss',
    'verbosity': -1,
    'random_state': SEED
})

print("\n--- 2. Building Advanced Base Models ---")

# Define comprehensive base models
base_models = {
    'lgb_advanced': lgb.LGBMClassifier(
        **{k: v for k, v in best_lgb_params.items() 
           if k not in ['drop_rate', 'max_drop', 'skip_drop']},
        n_estimators=2000
    ),
    
    'lgb_dart': lgb.LGBMClassifier(
        objective='binary',
        boosting_type='dart',
        num_leaves=100,
        learning_rate=0.02,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        drop_rate=0.1,
        max_drop=50,
        skip_drop=0.5,
        scale_pos_weight=neg_ratio / pos_ratio,
        n_estimators=2000,
        random_state=SEED,
        verbosity=-1
    ),
    
    'xgb_advanced': xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        learning_rate=0.02,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        colsample_bylevel=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=neg_ratio / pos_ratio,
        n_estimators=2000,
        random_state=SEED,
        verbosity=0
    ),
    
    'cat_advanced': CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=8,
        l2_leaf_reg=10.0,
        border_count=254,
        random_seed=SEED,
        verbose=False,
        eval_metric='F1',
        class_weights={0: 1, 1: 5},
        early_stopping_rounds=100
    ),
    
    'rf_advanced': RandomForestClassifier(
        n_estimators=1000,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight={0: 1, 1: 8},
        random_state=SEED,
        n_jobs=-1
    ),
    
    'et_advanced': ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight={0: 1, 1: 8},
        random_state=SEED,
        n_jobs=-1
    ),
    
    'gb_advanced': GradientBoostingClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=SEED
    )
}

print("\n--- 3. Level 1: Training Base Models ---")

# Efficient cross-validation
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# Store results
level1_predictions = {}
level1_test_predictions = {}
model_scores = {}

for model_name, model in base_models.items():
    print(f"\nTraining {model_name}...")
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    fold_scores = []
    
    fold_count = 0
    total_folds = N_SPLITS
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        fold_count += 1
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Model-specific training
        if model_name.startswith('lgb'):
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
        elif model_name.startswith('xgb'):
            model.fit(X_tr, y_tr)
        elif model_name.startswith('cat'):
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100,
                verbose=False
            )
        else:  # RF, ET, GB
            model.fit(X_tr, y_tr)
        
        # Predictions
        val_proba = model.predict_proba(X_val)[:, 1]
        
        # Find optimal threshold for F1
        best_f1 = 0
        best_thresh = 0.5
        for threshold in np.arange(0.1, 0.9, 0.01):
            f1 = f1_objective(y_val, val_proba, threshold)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = threshold
        
        fold_scores.append({'f1': best_f1, 'threshold': best_thresh})
        
        # Accumulate OOF predictions
        oof_preds[val_idx] = val_proba
        test_preds += model.predict_proba(X_test)[:, 1] / total_folds
        
        print(f"  Fold {fold_count}/{total_folds} completed")
    
    # Save results
    level1_predictions[model_name] = oof_preds
    level1_test_predictions[model_name] = test_preds
    model_scores[model_name] = fold_scores
    
    # Print average performance
    avg_f1 = np.mean([s['f1'] for s in fold_scores])
    print(f"{model_name}: Average F1={avg_f1:.4f}")

print("\n--- 4. Level 2: Advanced Meta-Learning ---")

# Prepare Level 1 features
meta_train = pd.DataFrame(level1_predictions)
meta_test = pd.DataFrame(level1_test_predictions)

# Scale features for neural network
scaler = StandardScaler()
meta_train_scaled = scaler.fit_transform(meta_train)
meta_test_scaled = scaler.transform(meta_test)

print("Level 1 features shape:", meta_train.shape)

# Define Level 2 meta-models
meta_models = {
    'logistic_ridge': LogisticRegression(
        C=0.1, 
        class_weight={0: 1, 1: 5},
        random_state=SEED,
        max_iter=2000
    ),
    
    'bayesian_ridge': BayesianRidge(
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6
    ),
    
    'elastic_net': ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        random_state=SEED,
        max_iter=2000
    ),
    
    'mlp_classifier': create_mlp_classifier()
}

# Train meta-models
level2_predictions = {}
level2_test_predictions = {}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

for meta_name, meta_model in meta_models.items():
    print(f"\nTraining meta-model: {meta_name}")
    
    if meta_name == 'mlp_classifier':
        # MLP Classifier Meta-model
        meta_oof = np.zeros(len(X_train))
        meta_test_preds = np.zeros(len(X_test))
        
        for train_idx, val_idx in skf.split(meta_train, y_train):
            # Create and train MLP
            mlp_model = create_mlp_classifier()
            
            # Train with scaled features
            mlp_model.fit(meta_train_scaled[train_idx], y_train.iloc[train_idx])
            
            # Predict
            meta_oof[val_idx] = mlp_model.predict_proba(meta_train_scaled[val_idx])[:, 1]
            meta_test_preds += mlp_model.predict_proba(meta_test_scaled)[:, 1] / 5
        
        level2_predictions[meta_name] = meta_oof
        level2_test_predictions[meta_name] = meta_test_preds
        
    else:
        # Traditional meta-models
        meta_oof = np.zeros(len(X_train))
        meta_test_preds = np.zeros(len(X_test))
        
        for train_idx, val_idx in skf.split(meta_train, y_train):
            if meta_name in ['bayesian_ridge', 'elastic_net']:
                # Regression-based meta-models
                meta_model.fit(meta_train.iloc[train_idx], y_train.iloc[train_idx])
                meta_oof[val_idx] = meta_model.predict(meta_train.iloc[val_idx])
                meta_test_preds += meta_model.predict(meta_test) / 5
            else:
                # Classification-based meta-models
                meta_model.fit(meta_train.iloc[train_idx], y_train.iloc[train_idx])
                meta_oof[val_idx] = meta_model.predict_proba(meta_train.iloc[val_idx])[:, 1]
                meta_test_preds += meta_model.predict_proba(meta_test)[:, 1] / 5
        
        level2_predictions[meta_name] = meta_oof
        level2_test_predictions[meta_name] = meta_test_preds

print("\n--- 5. Advanced Ensemble Weight Optimization ---")

def ensemble_objective(weights, predictions, y_true):
    """Objective function for ensemble weight optimization"""
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize weights
    
    ensemble_pred = np.zeros(len(y_true))
    for i, (name, pred) in enumerate(predictions.items()):
        ensemble_pred += weights[i] * pred
    
    # Find optimal threshold
    best_f1 = 0
    for threshold in np.arange(0.1, 0.9, 0.02):
        f1 = f1_objective(y_true, ensemble_pred, threshold)
        if f1 > best_f1:
            best_f1 = f1
    
    return -best_f1  # Minimize negative F1

# Optimize ensemble weights
n_models = len(level2_predictions)
initial_weights = np.ones(n_models) / n_models
bounds = [(0.0, 1.0) for _ in range(n_models)]
constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

result = minimize(
    ensemble_objective,
    initial_weights,
    args=(level2_predictions, y_train),
    method='SLSQP',
    bounds=bounds,
    constraints=constraint
)

optimal_weights = result.x
print(f"Optimal weights: {dict(zip(level2_predictions.keys(), optimal_weights))}")

# Create final ensemble
final_ensemble = np.zeros(len(X_train))
final_test_ensemble = np.zeros(len(X_test))

for i, (name, pred) in enumerate(level2_predictions.items()):
    final_ensemble += optimal_weights[i] * pred
    final_test_ensemble += optimal_weights[i] * level2_test_predictions[name]

print("\n--- 6. Final Threshold Optimization ---")

# Multi-threshold strategy
threshold_candidates = np.arange(0.05, 0.95, 0.005)
best_f1 = 0
best_threshold = 0.5

for threshold in threshold_candidates:
    f1 = f1_objective(y_train, final_ensemble, threshold)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Optimal threshold: {best_threshold:.3f}")
print(f"Best F1: {best_f1:.4f}")

# Final predictions
final_preds = (final_ensemble >= best_threshold).astype(int)

# Calculate final metrics
tn, fp, fn, tp = confusion_matrix(y_train, final_preds).ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nFinal Performance:")
print(f"F1: {best_f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Predicted positive rate: {final_preds.mean():.4f}")

print("\nFinal Classification Report:")
print(classification_report(y_train, final_preds))

print("\n--- 7. Saving Advanced Results ---")

# Save all results
np.save(RESULT_PATH + 'advanced_ensemble_test_preds.npy', final_test_ensemble)
np.save(RESULT_PATH + 'advanced_ensemble_oof_preds.npy', final_ensemble)

with open(RESULT_PATH + 'advanced_threshold.pkl', 'wb') as f:
    pickle.dump(best_threshold, f)

# Save comprehensive model info
advanced_model_info = {
    'best_f1': best_f1,
    'best_threshold': best_threshold,
    'optimal_weights': dict(zip(level2_predictions.keys(), optimal_weights)),
    'level1_scores': model_scores,
    'optuna_best_params': study.best_params,
    'optuna_best_value': study.best_value,
    'n_trials': 50,
    'ensemble_architecture': 'Multi-level Stacking + Weight Optimization'
}

with open(RESULT_PATH + 'advanced_model_info.pkl', 'wb') as f:
    pickle.dump(advanced_model_info, f)

print(f"\n✅ Advanced ensemble modeling completed!")
print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Target: Beat current best (Public F1: 0.5109)")
print(f"Expected: F1 > 0.52 with advanced multi-level ensemble") 