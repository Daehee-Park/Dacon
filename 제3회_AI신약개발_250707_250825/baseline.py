import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
import random

CFG = {
    'NBITS': 2048,
    'SEED': 42
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
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=CFG['NBITS'])
        return np.array(fp)
    else:
        return np.zeros((CFG['NBITS'],))

def IC50_to_pIC50(ic50_nM):
    ic50_nM = np.clip(ic50_nM, 1e-10, None)
    return 9 - np.log10(ic50_nM)

chembl = pd.read_csv("data/ChEMBL_ASK1(IC50).csv", sep=';')
pubchem = pd.read_csv("data/Pubchem_ASK1.csv")

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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=CFG['SEED'])

model = RandomForestRegressor(random_state=CFG['SEED'])
model.fit(X_train, y_train)

y_val_pred = model.predict(X_val)
rmse = mean_squared_error(IC50_to_pIC50(y_val), IC50_to_pIC50(y_val_pred), squared=False)
print(f"\n Validation RMSE (IC50 scale): {rmse:.4f}\n")

def pIC50_to_IC50(pIC50):
    return 10 ** (9 - pIC50)

test = pd.read_csv("data/test.csv")
test['Fingerprint'] = test['Smiles'].apply(smiles_to_fingerprint)
test = test[test['Fingerprint'].notnull()]

X_test = np.stack(test['Fingerprint'].values)
test['pIC50_pred'] = model.predict(X_test)
test['ASK1_IC50_nM'] = pIC50_to_IC50(test['pIC50_pred'])

submission = pd.read_csv('data/sample_submission.csv') 
submission['ASK1_IC50_nM'] = test['ASK1_IC50_nM']
submission.to_csv("baseline_submit.csv", index=False)