import pandas as pd
import numpy as np
import pickle
import os

print("=== Creating F1-Optimized Submission File ===")

# Paths
RESULT_PATH = './result/'

# Load test predictions and metadata
test_preds = np.load(RESULT_PATH + 'f1_optimized_test_preds.npy')
test_ids = pd.read_csv(RESULT_PATH + 'test_ids_enhanced.csv')

with open(RESULT_PATH + 'f1_optimized_threshold.pkl', 'rb') as f:
    optimal_threshold = pickle.load(f)

with open(RESULT_PATH + 'f1_model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

print(f"Test samples: {len(test_preds)}")
print(f"Test predictions shape: {test_preds.shape}")
print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Best ensemble method: {model_info['best_ensemble']}")
print(f"Expected F1 score: {model_info['best_f1']:.4f}")

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

# Compare with previous best submission if exists
prev_submission_path = '../submission02/result/submission_enhanced.csv'
if os.path.exists(prev_submission_path):
    prev_submission = pd.read_csv(prev_submission_path)
    
    # Compare prediction rates
    prev_cancer_rate = prev_submission['Cancer'].mean()
    current_cancer_rate = submission['Cancer'].mean()
    
    print(f"\nComparison with previous submission:")
    print(f"Previous cancer rate: {prev_cancer_rate:.4f}")
    print(f"Current cancer rate: {current_cancer_rate:.4f}")
    print(f"Rate change: {current_cancer_rate - prev_cancer_rate:+.4f}")
    
    # Agreement rate
    agreement = (prev_submission['Cancer'] == submission['Cancer']).mean()
    print(f"Agreement with previous: {agreement:.4f}")
else:
    print(f"\n⚠️ Previous submission not found for comparison")

# Display submission preview
print(f"\nSubmission preview:")
print(submission.head(6))

print(f"\nCancer predictions in first 100 samples: {submission.head(100)['Cancer'].sum()}")

# Save submission file
submission_path = RESULT_PATH + 'submission_f1_optimized.csv'
submission.to_csv(submission_path, index=False)
print(f"\nSubmission file saved: {submission_path}")

print(f"\n--- Final Submission Statistics ---")
print(f"Total test samples: {len(submission)}")
print(f"Predicted Cancer cases: {(submission['Cancer'] == 1).sum()}")
print(f"Predicted Benign cases: {(submission['Cancer'] == 0).sum()}")
print(f"Cancer prediction rate: {submission['Cancer'].mean():.4f}")

# Validate submission format
print(f"\n--- Format Validation ---")

# Check if columns match expected format
expected_columns = ['ID', 'Cancer']
print(f"Columns match: {list(submission.columns) == expected_columns}")

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
expected_shape = (46204, 2)  # Based on test set size
print(f"Shape correct: {submission.shape == expected_shape}")

if all([
    list(submission.columns) == expected_columns,
    id_format_correct,
    not has_missing,
    cancer_values_correct,
    submission.shape == expected_shape
]):
    print(f"✅ Submission format is perfect!")
else:
    print(f"❌ Submission format has issues!")

print(f"\n✅ F1-optimized submission ready for upload!")
print(f"Expected improvements:")
print(f"1. Enhanced feature engineering (58 features)")
print(f"2. F1-optimized hyperparameter tuning") 
print(f"3. Advanced ensemble strategies")
print(f"4. Precision-Recall balance optimization")
print(f"5. Target: Public F1 > 0.52") 