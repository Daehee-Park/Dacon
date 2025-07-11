# Dataset Information

## `train.csv` [File]
- `ID`: Unique identifier for each sample  
- `Age`: Age of the patient  
- `Gender`: Gender  
- `Country`: Nationality  
- `Race`: Ethnicity  
- `Family_Background`: Family history of disease  
- `Radiation_History`: History of radiation exposure  
- `Iodine_Deficiency`: Iodine deficiency status  
- `Smoke`: Smoking status  
- `Weight_Risk`: Weight-related health risk  
- `Diabetes`: Diabetes status  
- `Nodule_Size`: Thyroid nodule size  
- `TSH_Result`: TSH hormone test result  
- `T4_Result`: T4 hormone test result  
- `T3_Result`: T3 hormone test result  
- `Cancer`: Thyroid cancer status (0: Benign, 1: Malignant)  

## `test.csv` [File]
- `ID`: Unique identifier for each sample  
- Same features as `train.csv` excluding the `Cancer` column  

---

# Task

Develop an AI classification algorithm for **thyroid cancer diagnosis**.

---

# Description

Build a binary classification model that predicts whether a thyroid-related case is benign or malignant based on patient health data.

---

## Leaderboard

- **Evaluation Metric**: Binary F1 Score  
- **Public Score**: Based on 30% of the test data (pre-sampled)  
- **Private Score**: Based on the remaining 70% of the test data  

---

## External Data & Pre-trained Models

- Use of external data is **not allowed**.  
- Use of pre-trained models is allowed **only if**:
  - They are based on legally unrestricted sources
  - They are published in academic papers  

---

## AutoML Packages

- **All AutoML packages are prohibited.**