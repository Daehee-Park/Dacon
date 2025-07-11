import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from autogluon.core.metrics import make_scorer

def competition_score_func(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    대회 리더보드 산정 방식
    Score = 0.5 × (1 - min(A, 1)) + 0.5 × B
    A: Normalized RMSE 오차 (예측 정확도)
    B: 예측값과 실제값 간의 선형 상관관계 (변화 경향성 반영)
    """
    # A: Normalized RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_range = y_true.max() - y_true.min()
    normalized_rmse = rmse / y_range if y_range > 0 else 0
    A = normalized_rmse
    
    # B: Pearson correlation
    correlation, _ = pearsonr(y_true, y_pred)
    B = correlation if not np.isnan(correlation) else 0
    
    # Final score
    score = 0.5 * (1 - min(A, 1)) + 0.5 * B
    
    return score

# AutoGluon Scorer 생성
competition_scorer = make_scorer(
    name='competition_score',
    score_func=competition_score_func,
    optimum=1.0,  # 최적값: A=0, B=1일 때 score=1
    greater_is_better=True,  # 높을수록 좋음
    needs_pred=True  # regression 메트릭
) 