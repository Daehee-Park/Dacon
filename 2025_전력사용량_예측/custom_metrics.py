import numpy as np
from sklearn.metrics import make_scorer

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    표준 SMAPE (Symmetric Mean Absolute Percentage Error) 구현
    \[
    \text{SMAPE} = \frac{100\%}{n} \sum_{t=1}^{n} \frac{2 \times |\hat{y}_t - y_t|}{|\hat{y}_t| + |y_t|}
    \]
    """
    # 분모가 0이 되는 경우 방지 (|y_true| + |y_pred| = 0)
    denominator = np.abs(y_true) + np.abs(y_pred)
    # 분모가 0인 경우는 실제값과 예측값이 모두 0인 경우이므로 오차 0으로 처리
    safe_denominator = np.where(denominator == 0, 1, denominator)
    
    # 표준 SMAPE 계산
    smape_vals = 2 * np.abs(y_pred - y_true) / safe_denominator
    return 100 * np.mean(smape_vals)

# Scorer 생성
smape_scorer = make_scorer(
    name='smape',
    score_func=smape,
    greater_is_better=False,
    needs_proba=False,
    needs_threshold=False
)

# AutoGluon 호환 Scorer (옵션)
try:
    from autogluon.core.metrics import make_scorer as ag_make_scorer
    ag_smape_scorer = ag_make_scorer(
        name='smape',
        score_func=smape,
        optimum=0.0,  # 최적값: 0
        greater_is_better=False
    )
except ImportError:
    ag_smape_scorer = None
    print("AutoGluon not available. Using standard sklearn scorer.") 