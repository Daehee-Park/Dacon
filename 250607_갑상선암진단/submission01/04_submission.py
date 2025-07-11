import pandas as pd
import numpy as np
import pickle
import warnings

warnings.filterwarnings('ignore')

# 데이터 경로 설정
DATA_PATH = './data/'

print("--- Creating Submission File ---")

# 1. Test IDs 불러오기
test_ids = pd.read_csv(DATA_PATH + 'test_ids.csv')['ID']
print(f"Test samples: {len(test_ids)}")

# 2. Ensemble 예측값 불러오기
ensemble_test_preds = np.load(DATA_PATH + 'ensemble_test_preds.npy')
print(f"Test predictions shape: {ensemble_test_preds.shape}")

# 3. 최적 threshold 불러오기
with open(DATA_PATH + 'ensemble_threshold.pkl', 'rb') as f:
    optimal_threshold = pickle.load(f)
print(f"Optimal threshold: {optimal_threshold:.3f}")

# 4. 확률값을 0/1 예측으로 변환
test_predictions = (ensemble_test_preds >= optimal_threshold).astype(int)

# 5. 예측 결과 분포 확인
print(f"Test predictions distribution:")
print(f"Class 0: {(test_predictions == 0).sum()}")
print(f"Class 1: {(test_predictions == 1).sum()}")
print(f"Positive ratio: {test_predictions.mean():.4f}")

# 6. Submission 파일 생성
submission_df = pd.DataFrame({
    'ID': test_ids,
    'Cancer': test_predictions
})

print(f"\nSubmission file preview:")
print(submission_df.head(10))
print(f"Submission shape: {submission_df.shape}")

# 7. 파일 저장
submission_path = DATA_PATH + 'submission.csv'
submission_df.to_csv(submission_path, index=False)

print(f"\nSubmission file saved: {submission_path}")

# 8. 기본 통계 출력
print(f"\n--- Submission Statistics ---")
print(f"Total test samples: {len(submission_df)}")
print(f"Predicted Cancer cases: {submission_df['Cancer'].sum()}")
print(f"Predicted Benign cases: {(submission_df['Cancer'] == 0).sum()}")
print(f"Cancer prediction rate: {submission_df['Cancer'].mean():.4f}")

# 9. 검증 - 샘플 submission과 형식 비교
try:
    sample_submission = pd.read_csv(DATA_PATH + 'sample_submission.csv')
    print(f"\n--- Format Validation ---")
    print(f"Sample submission shape: {sample_submission.shape}")
    print(f"Our submission shape: {submission_df.shape}")
    print(f"Columns match: {list(sample_submission.columns) == list(submission_df.columns)}")
    print(f"ID order match: {(sample_submission['ID'] == submission_df['ID']).all()}")
    
    if len(sample_submission) == len(submission_df):
        print("✅ Submission format is correct!")
    else:
        print("❌ Submission length mismatch!")
        
except FileNotFoundError:
    print("⚠️ Sample submission file not found - skipping format validation")

print("\n--- Ready for submission! ---") 