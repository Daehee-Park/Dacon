import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

DATA_PATH = '../data/'
RESULT_PATH = './result/'

print("=== Creating Enhanced Submission File ===")

# 1. Test IDs 불러오기
test_ids = pd.read_csv(RESULT_PATH + 'test_ids_enhanced.csv')['ID']
print(f"Test samples: {len(test_ids)}")

# 2. Stacking 예측값 불러오기
stacking_test_preds = np.load(RESULT_PATH + 'stacking_test_preds.npy')
print(f"Test predictions shape: {stacking_test_preds.shape}")

# 3. 최적 threshold 불러오기
with open(RESULT_PATH + 'stacking_threshold.pkl', 'rb') as f:
    optimal_threshold = pickle.load(f)
print(f"Optimal threshold: {optimal_threshold:.3f}")

# 4. 확률값을 0/1 예측으로 변환
test_predictions = (stacking_test_preds >= optimal_threshold).astype(int)

# 5. 예측 결과 분포 확인
print(f"\nTest predictions distribution:")
print(f"Class 0 (Benign): {(test_predictions == 0).sum()}")
print(f"Class 1 (Cancer): {(test_predictions == 1).sum()}")
print(f"Cancer prediction rate: {test_predictions.mean():.4f}")

# 6. 이전 submission과 비교
try:
    previous_submission = pd.read_csv(DATA_PATH + 'submission.csv')
    previous_cancer_count = previous_submission['Cancer'].sum()
    current_cancer_count = test_predictions.sum()
    
    print(f"\n--- Comparison with Previous Submission ---")
    print(f"Previous cancer predictions: {previous_cancer_count} ({previous_cancer_count/len(test_ids)*100:.1f}%)")
    print(f"Current cancer predictions: {current_cancer_count} ({current_cancer_count/len(test_ids)*100:.1f}%)")
    print(f"Difference: {current_cancer_count - previous_cancer_count} ({(current_cancer_count - previous_cancer_count)/len(test_ids)*100:.2f}%)")
    
    # 예측이 바뀐 샘플 수
    changed_predictions = (previous_submission['Cancer'].values != test_predictions).sum()
    print(f"Changed predictions: {changed_predictions} ({changed_predictions/len(test_ids)*100:.1f}%)")
    
except FileNotFoundError:
    print("\n⚠️ Previous submission not found for comparison")

# 7. Submission 파일 생성
submission_df = pd.DataFrame({
    'ID': test_ids,
    'Cancer': test_predictions
})

print(f"\nSubmission preview:")
print(submission_df.head(10))
print(f"\nCancer predictions in first 100 samples: {submission_df.head(100)['Cancer'].sum()}")

# 8. 파일 저장
submission_path = RESULT_PATH + 'submission_enhanced.csv'
submission_df.to_csv(submission_path, index=False)

print(f"\nSubmission file saved: {submission_path}")

# 9. 최종 통계
print(f"\n--- Final Submission Statistics ---")
print(f"Total test samples: {len(submission_df)}")
print(f"Predicted Cancer cases: {submission_df['Cancer'].sum()}")
print(f"Predicted Benign cases: {(submission_df['Cancer'] == 0).sum()}")
print(f"Cancer prediction rate: {submission_df['Cancer'].mean():.4f}")

# 10. 형식 검증
try:
    sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')
    print(f"\n--- Format Validation ---")
    print(f"Columns match: {list(sample_submission.columns) == list(submission_df.columns)}")
    print(f"ID order match: {(sample_submission['ID'] == submission_df['ID']).all()}")
    print(f"Shape match: {sample_submission.shape == submission_df.shape}")
    
    if all([
        list(sample_submission.columns) == list(submission_df.columns),
        (sample_submission['ID'] == submission_df['ID']).all(),
        sample_submission.shape == submission_df.shape
    ]):
        print("✅ Submission format is perfect!")
    else:
        print("❌ Format issue detected!")
        
except FileNotFoundError:
    print("⚠️ Sample submission file not found - skipping validation")

print("\n✅ Enhanced submission ready for upload!")
print("Expected improvements:")
print("1. Higher recall (catching more cancer cases)")
print("2. Reduced false negative rate")
print("3. Better feature engineering")
print("4. Stacking ensemble for robustness") 