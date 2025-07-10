import pandas as pd
import numpy as np
import pickle
import os

print("=== Creating Advanced Ensemble Submission File ===")

# Paths
RESULT_PATH = './result/'

# Load advanced ensemble predictions and metadata
test_preds = np.load(RESULT_PATH + 'advanced_ensemble_test_preds.npy')
test_ids = pd.read_csv(RESULT_PATH + 'test_ids_enhanced.csv')

with open(RESULT_PATH + 'advanced_threshold.pkl', 'rb') as f:
    optimal_threshold = pickle.load(f)

with open(RESULT_PATH + 'advanced_model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

print(f"Test samples: {len(test_preds)}")
print(f"Test predictions shape: {test_preds.shape}")
print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Expected F1 score: {model_info['best_f1']:.4f}")
print(f"Ensemble architecture: {model_info['ensemble_architecture']}")
print(f"Optuna trials completed: {model_info['n_trials']}")
print(f"Optuna best value: {model_info['optuna_best_value']:.4f}")

# Display ensemble weights
print(f"\nOptimal ensemble weights:")
for name, weight in model_info['optimal_weights'].items():
    print(f"  {name}: {weight:.4f}")

# Apply threshold to get binary predictions
binary_preds = (test_preds >= optimal_threshold).astype(int)

print(f"\nTest predictions distribution:")
print(f"Class 0 (Benign): {(binary_preds == 0).sum()}")
print(f"Class 1 (Cancer): {(binary_preds == 1).sum()}")
print(f"Cancer prediction rate: {binary_preds.mean():.4f}")

# Create submission DataFrame
submission = pd.DataFrame({
    'ID': test_ids['ID'],
    'Cancer': binary_preds
})

# Compare with previous submissions
comparison_files = [
    ('../submission05/result/submission_f1_optimized.csv', '5th attempt (F1-optimized)'),
    ('../submission02/result/submission_enhanced.csv', '4th attempt (Cost-sensitive)')
]

for file_path, label in comparison_files:
    if os.path.exists(file_path):
        prev_submission = pd.read_csv(file_path)
        
        # Compare prediction rates
        prev_cancer_rate = prev_submission['Cancer'].mean()
        current_cancer_rate = submission['Cancer'].mean()
        
        print(f"\nComparison with {label}:")
        print(f"Previous cancer rate: {prev_cancer_rate:.4f}")
        print(f"Current cancer rate: {current_cancer_rate:.4f}")
        print(f"Rate change: {current_cancer_rate - prev_cancer_rate:+.4f}")
        
        # Agreement rate
        agreement = (prev_submission['Cancer'] == submission['Cancer']).mean()
        print(f"Agreement: {agreement:.4f}")
        
        # Different predictions analysis
        diff_count = (prev_submission['Cancer'] != submission['Cancer']).sum()
        print(f"Different predictions: {diff_count} samples ({diff_count/len(submission)*100:.1f}%)")

# Display submission preview
print(f"\nSubmission preview:")
print(submission.head(10))

print(f"\nCancer predictions in first 100 samples: {submission.head(100)['Cancer'].sum()}")

# Save submission file
submission_path = RESULT_PATH + 'submission_advanced_ensemble.csv'
submission.to_csv(submission_path, index=False)
print(f"\nSubmission file saved: {submission_path}")

print(f"\n--- Final Submission Statistics ---")
print(f"Total test samples: {len(submission)}")
print(f"Predicted Cancer cases: {(submission['Cancer'] == 1).sum()}")
print(f"Predicted Benign cases: {(submission['Cancer'] == 0).sum()}")
print(f"Cancer prediction rate: {submission['Cancer'].mean():.4f}")

# Advanced validation
print(f"\n--- Advanced Format Validation ---")

# Check if columns match expected format
expected_columns = ['ID', 'Cancer']
columns_correct = list(submission.columns) == expected_columns
print(f"Columns match: {columns_correct}")

# Check ID format
id_format_correct = all(submission['ID'].str.startswith('TEST_'))
print(f"ID format correct: {id_format_correct}")

# Check for missing values
has_missing = submission.isnull().any().any()
print(f"No missing values: {not has_missing}")

# Check Cancer column values
cancer_values_correct = set(submission['Cancer'].unique()) <= {0, 1}
print(f"Cancer values correct (0/1): {cancer_values_correct}")

# Check submission shape
expected_shape = (46204, 2)
shape_correct = submission.shape == expected_shape
print(f"Shape correct: {shape_correct}")

# Check for duplicated IDs
no_duplicate_ids = not submission['ID'].duplicated().any()
print(f"No duplicate IDs: {no_duplicate_ids}")

# Statistical validation
cancer_rate_reasonable = 0.05 <= submission['Cancer'].mean() <= 0.25
print(f"Cancer rate reasonable (5-25%): {cancer_rate_reasonable}")

# Overall validation
all_checks_passed = all([
    columns_correct,
    id_format_correct,
    not has_missing,
    cancer_values_correct,
    shape_correct,
    no_duplicate_ids,
    cancer_rate_reasonable
])

if all_checks_passed:
    print(f"✅ Submission format is perfect!")
else:
    print(f"❌ Submission format has issues!")

print(f"\n--- Model Architecture Summary ---")
print(f"Level 1 models: 7 advanced base models")
print(f"- LightGBM (Optuna-tuned): {model_info['optuna_best_value']:.4f}")
print(f"- LightGBM DART, XGBoost, CatBoost")
print(f"- RandomForest, ExtraTrees, GradientBoosting")
print(f"Level 2 meta-models: 4 sophisticated ensemble methods")
print(f"- Logistic Regression (Ridge)")
print(f"- Bayesian Ridge Regression")
print(f"- Elastic Net Regression")
print(f"- Neural Network (3-layer)")
print(f"Optimization: Scipy weight optimization")
print(f"Cross-validation: 10-fold × 2 repeats = 20 folds")

print(f"\n✅ Advanced ensemble submission ready for upload!")
print(f"Expected breakthrough improvements:")
print(f"1. 100-trial Optuna optimization with pruning")
print(f"2. Multi-level stacking architecture")
print(f"3. Neural network meta-learning")
print(f"4. Bayesian & Elastic Net regularization")
print(f"5. Advanced weight optimization")
print(f"6. Robust 20-fold cross-validation")
print(f"7. Target: Public F1 > 0.52 (performance breakthrough)") 