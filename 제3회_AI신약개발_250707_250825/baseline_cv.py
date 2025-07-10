import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import os
import random

CFG = {
    'NBITS': 2048,
    'SEED': 42,
    'N_FOLDS': 5
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(CFG['SEED'])

# SMILES 데이터를 분자 지문으로 변환
def smiles_to_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        # 새로운 MorganGenerator 사용
        morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=CFG['NBITS'])
        fp = morgan_gen.GetFingerprintAsNumPy(mol)
        return fp
    else:
        return np.zeros((CFG['NBITS'],))

def IC50_to_pIC50(ic50_nM):
    ic50_nM = np.clip(ic50_nM, 1e-10, None)
    return 9 - np.log10(ic50_nM)

def pIC50_to_IC50(pIC50):
    return 10 ** (9 - pIC50)

def calculate_normalized_rmse(y_true_ic50, y_pred_ic50):
    """Calculate normalized RMSE on IC50 scale"""
    rmse = np.sqrt(mean_squared_error(y_true_ic50, y_pred_ic50))
    # Normalize by the range of true values
    y_range = np.max(y_true_ic50) - np.min(y_true_ic50)
    return rmse / y_range

def calculate_correlation(y_true_pic50, y_pred_pic50):
    """Calculate correlation coefficient squared"""
    corr, _ = pearsonr(y_true_pic50, y_pred_pic50)
    return corr ** 2

def calculate_final_score(normalized_rmse, correlation_squared):
    """Calculate final score according to competition rules"""
    A = normalized_rmse
    B = correlation_squared
    score = 0.4 * (1 - min(A, 1)) + 0.6 * B
    return score

# 데이터 로드 및 전처리
chembl = pd.read_csv("data/ChEMBL_ASK1(IC50).csv", sep=';')
pubchem = pd.read_csv("data/Pubchem_ASK1.csv", low_memory=False)

chembl.columns = chembl.columns.str.strip().str.replace('"', '')
chembl = chembl[chembl['Standard Type'] == 'IC50']
chembl = chembl[['Smiles', 'Standard Value']].rename(columns={'Smiles': 'smiles', 'Standard Value': 'ic50_nM'}).dropna()
chembl['ic50_nM'] = pd.to_numeric(chembl['ic50_nM'], errors='coerce')
chembl['pIC50'] = IC50_to_pIC50(chembl['ic50_nM'])

pubchem = pubchem[['SMILES', 'Activity_Value']].rename(columns={'SMILES': 'smiles', 'Activity_Value': 'ic50_nM'}).dropna()
pubchem['ic50_nM'] = pd.to_numeric(pubchem['ic50_nM'], errors='coerce')
pubchem['pIC50'] = IC50_to_pIC50(pubchem['ic50_nM'])

total = pd.concat([chembl, pubchem], ignore_index=True)
total = total.drop_duplicates(subset='smiles')
total = total[total['ic50_nM'] > 0].dropna()

total['Fingerprint'] = total['smiles'].apply(smiles_to_fingerprint)
total = total[total['Fingerprint'].notnull()]
X = np.stack(total['Fingerprint'].values)
y = total['pIC50'].values
y_ic50 = total['ic50_nM'].values

# Cross-validation
kf = KFold(n_splits=CFG['N_FOLDS'], shuffle=True, random_state=CFG['SEED'])

cv_scores = []
cv_rmse_scores = []
cv_corr_scores = []
cv_final_scores = []

print("Starting Cross-Validation...")
print("=" * 50)

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}/{CFG['N_FOLDS']}")
    
    # Split data
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    y_ic50_train, y_ic50_val = y_ic50[train_idx], y_ic50[val_idx]
    
    # Train model
    model = RandomForestRegressor(random_state=CFG['SEED'])
    model.fit(X_train, y_train)
    
    # Predict
    y_val_pred = model.predict(X_val)
    y_ic50_val_pred = pIC50_to_IC50(y_val_pred)
    
    # Calculate metrics
    normalized_rmse = calculate_normalized_rmse(y_ic50_val, y_ic50_val_pred)
    correlation_squared = calculate_correlation(y_val, y_val_pred)
    final_score = calculate_final_score(normalized_rmse, correlation_squared)
    
    # Store results
    cv_rmse_scores.append(normalized_rmse)
    cv_corr_scores.append(correlation_squared)
    cv_final_scores.append(final_score)
    
    print(f"  Normalized RMSE: {normalized_rmse:.4f}")
    print(f"  Correlation²: {correlation_squared:.4f}")
    print(f"  Final Score: {final_score:.4f}")
    print()

# Print final results
print("=" * 50)
print("Cross-Validation Results:")
print(f"Average Normalized RMSE: {np.mean(cv_rmse_scores):.4f} ± {np.std(cv_rmse_scores):.4f}")
print(f"Average Correlation²: {np.mean(cv_corr_scores):.4f} ± {np.std(cv_corr_scores):.4f}")
print(f"Average Final Score: {np.mean(cv_final_scores):.4f} ± {np.std(cv_final_scores):.4f}")
print("=" * 50) 

# Train final model on all data and generate submission
print("Training final model for submission...")
final_model = RandomForestRegressor(random_state=CFG['SEED'])
final_model.fit(X, y)

# Load test data
test_df = pd.read_csv("data/test.csv")
test_df['Fingerprint'] = test_df['Smiles'].apply(smiles_to_fingerprint)
test_df = test_df[test_df['Fingerprint'].notnull()]
X_test = np.stack(test_df['Fingerprint'].values)

# Make predictions
test_predictions_pic50 = final_model.predict(X_test)
test_predictions_ic50 = pIC50_to_IC50(test_predictions_pic50)

# Create submission file
submission = pd.DataFrame({
    'ID': test_df['ID'],
    'ASK1_IC50_nM': test_predictions_ic50
})

submission.to_csv('output/baseline_cv_submission.csv', index=False)
print(f"Submission file saved: baseline_cv_submission.csv")
print(f"Number of predictions: {len(submission)}") 