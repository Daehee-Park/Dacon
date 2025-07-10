# Thyroid Cancer Classification Project Plan

## 1. Deeper Exploratory Data Analysis (EDA) & Preprocessing (`02_eda_and_preprocessing.py`)

- **Objective**: Understand feature distributions, relationships with the target variable, and prepare the data for modeling.
- **Steps**:
    1.  **Visualize Distributions**:
        -   Plot histograms and KDE plots for numerical features (`Age`, `Nodule_Size`, hormone results) to see their distributions, stratified by the `Cancer` target variable.
        -   Plot bar charts for categorical features to see their frequencies and relationship with the `Cancer` target.
    2.  **Correlation Analysis**:
        -   Create a correlation heatmap for numerical features to check for multicollinearity.
    3.  **Feature Encoding**:
        -   Apply **Label Encoding** to binary categorical features (e.g., `Gender`, `Smoke`, `Diabetes`, etc.).
        -   Apply **One-Hot Encoding** to multi-class categorical features (`Country`, `Race`).
    4.  **Feature Scaling**:
        -   Scale numerical features using `StandardScaler` to normalize their ranges. This is important for many models.
    5.  **Save Processed Data**:
        -   Save the processed training and testing data to be used in the modeling phase.

## 2. Model Training and Hyperparameter Tuning (`03_modeling.py`)

- **Objective**: Train several models, tune their hyperparameters, and select the best one based on the F1 score.
- **Steps**:
    1.  **Data Splitting**:
        -   Split the training data into a training set and a validation set.
        -   Use **stratified splitting** (`stratify=y`) to ensure the same class distribution in both sets, which is crucial due to the target imbalance.
    2.  **Model Selection**:
        -   Start with a baseline model like `LogisticRegression`.
        -   Train more advanced and powerful models like `LGBMClassifier` and `XGBClassifier`, which are typically very effective for tabular data.
        -   Address class imbalance using model parameters like `scale_pos_weight` (for XGBoost/LGBM) or `class_weight='balanced'`.
    3.  **Hyperparameter Tuning**:
        -   Use a systematic approach like `Optuna` or `GridSearchCV` to find the optimal hyperparameters for the best-performing model. The optimization metric will be the **F1 score**.
    4.  **Cross-Validation**:
        -   Employ `StratifiedKFold` cross-validation during training and tuning to get a more robust estimate of the model's performance.

## 3. Final Prediction and Submission (`04_inference.py`)

- **Objective**: Train the final model on all available training data and generate the submission file.
- **Steps**:
    1.  **Final Model Training**:
        -   Train the best model with the optimal hyperparameters on the **entire** training dataset.
    2.  **Inference**:
        -   Make predictions on the preprocessed test data.
    3.  **Submission File Generation**:
        -   Create `submission.csv` with `ID` and `Cancer` columns in the required format.