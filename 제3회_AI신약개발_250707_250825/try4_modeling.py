import os
import json
import pickle
import random
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna


CFG: Dict[str, int] = {
    'SEED': 33,
    'N_SPLITS': 5,
    'N_REPEATS': 2,
    'CPUS': 64,
    'N_TRIALS_L1': 60,
    'N_TRIALS_META': 60,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


seed_everything(CFG['SEED'])

OUTPUT_DIR = "./output/try4"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def pIC50_to_IC50(pIC50: np.ndarray) -> np.ndarray:
    return 10 ** (9 - pIC50)


def get_score(y_true_ic50, y_pred_ic50, y_true_pic50, y_pred_pic50):
    rmse = mean_squared_error(y_true_ic50, y_pred_ic50) ** 0.5
    nrmse = rmse / (np.max(y_true_ic50) - np.min(y_true_ic50))
    A = 1 - min(nrmse, 1)
    B = r2_score(y_true_pic50, y_pred_pic50)
    return 0.4 * A + 0.6 * B


def load_preprocessed() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    X_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'X_train.csv')).values.astype(np.float32)
    y_train = pd.read_csv(os.path.join(OUTPUT_DIR, 'y_train.csv'))['pIC50'].values.astype(np.float32)
    X_test = pd.read_csv(os.path.join(OUTPUT_DIR, 'X_test.csv')).values.astype(np.float32)
    test_ids = pd.read_csv(os.path.join(OUTPUT_DIR, 'test_ids.csv'))['ID'].values
    feature_info = json.load(open(os.path.join(OUTPUT_DIR, 'feature_info.json'), 'r', encoding='utf-8'))
    return X_train, y_train, X_test, test_ids, feature_info


def train_cv_model(model_name: str, params: Dict, X: np.ndarray, y: np.ndarray) -> Tuple[List, np.ndarray, float]:
    y_bins = pd.qcut(y, q=CFG['N_SPLITS'], labels=False, duplicates='drop')
    rskf = RepeatedStratifiedKFold(n_splits=CFG['N_SPLITS'], n_repeats=CFG['N_REPEATS'], random_state=CFG['SEED'])

    oof = np.zeros(len(X), dtype=np.float32)
    models = []
    fold_scores = []

    for fold, (trn_idx, val_idx) in enumerate(rskf.split(X, y_bins)):
        X_tr, X_va = X[trn_idx], X[val_idx]
        y_tr, y_va = y[trn_idx], y[val_idx]

        if model_name == 'lgb':
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(200, verbose=False)]
            )
        elif model_name == 'xgb':
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                verbose=False
            )
        elif model_name == 'cb':
            model = cb.CatBoostRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                early_stopping_rounds=200,
                verbose=False
            )
        else:
            raise ValueError('Unknown model name')

        pred_va = model.predict(X_va)
        oof[val_idx] = pred_va
        models.append(model)

        # record per-fold score
        fold_score = get_score(pIC50_to_IC50(y_va), pIC50_to_IC50(pred_va), y_va, pred_va)
        fold_scores.append(fold_score)

    y_ic50_true = pIC50_to_IC50(y)
    oof_ic50_pred = pIC50_to_IC50(oof)
    cv_score = get_score(y_ic50_true, oof_ic50_pred, y, oof)
    print(f"    {model_name.upper()} CV: {cv_score:.5f} (±{np.std(fold_scores):.5f}) over {len(fold_scores)} folds")
    return models, oof, cv_score


def optimize_blend(oof_dict: Dict[str, np.ndarray], y: np.ndarray) -> Tuple[Dict[str, float], float]:
    names = list(oof_dict.keys())
    init = np.ones(len(names), dtype=np.float64) / len(names)

    def objective(w):
        pred = np.zeros_like(y)
        for i, n in enumerate(names):
            pred += w[i] * oof_dict[n]
        score = get_score(pIC50_to_IC50(y), pIC50_to_IC50(pred), y, pred)
        return -score

    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bounds = [(0.0, 1.0)] * len(names)
    res = minimize(objective, init, method='SLSQP', bounds=bounds, constraints=cons)
    weights = {n: float(w) for n, w in zip(names, res.x)}
    return weights, -float(res.fun)


def tune_model(model_name: str, X: np.ndarray, y: np.ndarray) -> Dict:
    def objective(trial):
        if model_name == 'lgb':
            params = {
                'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'verbose': -1,
                'n_jobs': CFG['CPUS'], 'seed': CFG['SEED'], 'n_estimators': trial.suggest_int('n_estimators', 1500, 6000),
                'num_leaves': trial.suggest_int('num_leaves', 31, 128),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50)
            }
        elif model_name == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 1500, 6000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
                'max_depth': trial.suggest_int('max_depth', 5, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
                'random_state': CFG['SEED'], 'n_jobs': CFG['CPUS'], 'verbosity': 0, 'tree_method': 'hist'
            }
        else:  # cb
            params = {
                'iterations': trial.suggest_int('iterations', 1500, 6000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.08, log=True),
                'depth': trial.suggest_int('depth', 5, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                'random_seed': CFG['SEED'], 'thread_count': CFG['CPUS'], 'verbose': False
            }

        models, oof, cv = train_cv_model(model_name, params, X, y)
        return cv

    study = optuna.create_study(direction='maximize', study_name=f'try4_{model_name}')
    study.optimize(objective, n_trials=CFG['N_TRIALS_L1'], n_jobs=CFG['CPUS'])
    return study.best_params


def main():
    print("[try4_modeling] Loading preprocessed data...")
    X, y, X_test, test_ids, feature_info = load_preprocessed()
    print(f"  X: {X.shape}, X_test: {X_test.shape}")

    # Tune Level-1 models with Optuna
    print("[try4_modeling] Tuning Level-1 models...")
    lgb_best = tune_model('lgb', X, y)
    xgb_best = tune_model('xgb', X, y)
    cb_best = tune_model('cb', X, y)

    # Re-train Level-1 models using best params
    print("[try4_modeling] Training Level-1 models with best params...")
    lgb_params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'verbose': -1, 'n_jobs': CFG['CPUS'], 'seed': CFG['SEED']}
    lgb_params.update(lgb_best)
    xgb_params = {'random_state': CFG['SEED'], 'n_jobs': CFG['CPUS'], 'verbosity': 0, 'tree_method': 'hist'}
    xgb_params.update(xgb_best)
    cb_params = {'random_seed': CFG['SEED'], 'thread_count': CFG['CPUS'], 'verbose': False}
    cb_params.update(cb_best)

    lgb_models, lgb_oof, lgb_cv = train_cv_model('lgb', lgb_params, X, y)
    xgb_models, xgb_oof, xgb_cv = train_cv_model('xgb', xgb_params, X, y)
    cb_models, cb_oof, cb_cv = train_cv_model('cb', cb_params, X, y)

    oof_dict = {'lgb': lgb_oof, 'xgb': xgb_oof, 'cb': cb_oof}

    print("[try4_modeling] Optimizing blend weights...")
    weights, blend_cv = optimize_blend(oof_dict, y)
    print(f"  Weights: {weights}")
    print(f"  Blend CV: {blend_cv:.5f}")

    # Level-2 meta model tuning and CV stacking
    print("[try4_modeling] Training Level-2 meta-model with CV...")
    X_meta = np.column_stack([lgb_oof, xgb_oof, cb_oof]).astype(np.float32)
    y_bins = pd.qcut(y, q=CFG['N_SPLITS'], labels=False, duplicates='drop')

    def meta_objective(trial):
        params = {
            'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'verbose': -1,
            'n_jobs': CFG['CPUS'], 'seed': CFG['SEED'], 'n_estimators': trial.suggest_int('n_estimators', 1000, 4000),
            'num_leaves': trial.suggest_int('num_leaves', 15, 64),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50)
        }
        skf = StratifiedKFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
        oof_meta = np.zeros(len(X_meta), dtype=np.float32)
        for trn_idx, val_idx in skf.split(X_meta, y_bins):
            model = lgb.LGBMRegressor(**params)
            model.fit(X_meta[trn_idx], y[trn_idx], eval_set=[(X_meta[val_idx], y[val_idx])], eval_metric='rmse', callbacks=[lgb.early_stopping(200, verbose=False)])
            oof_meta[val_idx] = model.predict(X_meta[val_idx])
        return get_score(pIC50_to_IC50(y), pIC50_to_IC50(oof_meta), y, oof_meta)

    study_meta = optuna.create_study(direction='maximize', study_name='try4_meta')
    study_meta.optimize(meta_objective, n_trials=CFG['N_TRIALS_META'], n_jobs=CFG['CPUS'])
    meta_params = {'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt', 'verbose': -1, 'n_jobs': CFG['CPUS'], 'seed': CFG['SEED']}
    meta_params.update(study_meta.best_params)

    # Train meta with CV and compute OOF
    skf = StratifiedKFold(n_splits=CFG['N_SPLITS'], shuffle=True, random_state=CFG['SEED'])
    oof_meta = np.zeros(len(X_meta), dtype=np.float32)
    meta_models = []
    for trn_idx, val_idx in skf.split(X_meta, y_bins):
        m = lgb.LGBMRegressor(**meta_params)
        m.fit(X_meta[trn_idx], y[trn_idx])
        oof_meta[val_idx] = m.predict(X_meta[val_idx])
        meta_models.append(m)
    meta_cv = get_score(pIC50_to_IC50(y), pIC50_to_IC50(oof_meta), y, oof_meta)
    print(f"  Meta CV (OOF): {meta_cv:.5f}")

    # Save model artifacts (optional lightweight)
    with open(os.path.join(OUTPUT_DIR, 'level1_models.pkl'), 'wb') as f:
        pickle.dump({'lgb': lgb_models, 'xgb': xgb_models, 'cb': cb_models, 'weights': weights}, f)

    # Predict test
    print("[try4_modeling] Inference on test...")
    def avg_predict(models, X_):
        m_pred = np.zeros(X_.shape[0], dtype=np.float32)
        for m in models:
            m_pred += m.predict(X_) / len(models)
        return m_pred

    preds_lgb = avg_predict(lgb_models, X_test)
    preds_xgb = avg_predict(xgb_models, X_test)
    preds_cb = avg_predict(cb_models, X_test)

    # Create meta-level test features and predict
    X_meta_test = np.column_stack([preds_lgb, preds_xgb, preds_cb]).astype(np.float32)
    preds_test_meta = np.zeros(X_meta_test.shape[0], dtype=np.float32)
    for m in meta_models:
        preds_test_meta += m.predict(X_meta_test) / len(meta_models)

    # Convert to IC50
    preds_ic50 = pIC50_to_IC50(preds_test_meta)

    # Build submission, fill missing with train median in IC50 space
    print("[try4_modeling] Building submission...")
    sample = pd.read_csv('./data/sample_submission.csv')
    sub = sample[['ID']].merge(
        pd.DataFrame({'ID': test_ids, 'ASK1_IC50_nM': preds_ic50}), on='ID', how='left'
    )

    # Use robust median fallback. Load raw training IC50s if available
    try:
        train_raw = pd.read_csv('./data/ChEMBL_ASK1(IC50).csv', sep=';')
        train_pub = pd.read_csv('./data/Pubchem_ASK1.csv', low_memory=False)
        train_raw.columns = train_raw.columns.str.strip().str.replace('"', '')
        train_raw = train_raw[train_raw['Standard Type'] == 'IC50']
        train_raw = train_raw[['Standard Value']].rename(columns={'Standard Value': 'ic50_nM'})
        train_pub = train_pub[['Activity_Value']].rename(columns={'Activity_Value': 'ic50_nM'})
        ic50_all = pd.concat([train_raw, train_pub], ignore_index=True)['ic50_nM']
        ic50_all = pd.to_numeric(ic50_all, errors='coerce')
        ic50_all = ic50_all[ic50_all > 0]
        fallback = float(ic50_all.median())
    except Exception:
        fallback = float(np.median(pIC50_to_IC50(y)))

    sub['ASK1_IC50_nM'].fillna(fallback, inplace=True)

    out_path = os.path.join(OUTPUT_DIR, 'submission.csv')
    sub.to_csv(out_path, index=False)
    print(f"[try4_modeling] Saved submission to {out_path}")

    # Optional: submit via helper if available
    try:
        from dacon_submit import dacon_submit
        dacon_submit(submission_path=out_path, memo=f"try4 stacked blend, CV {blend_cv:.6f}")
    except Exception:
        pass


if __name__ == '__main__':
    main()


