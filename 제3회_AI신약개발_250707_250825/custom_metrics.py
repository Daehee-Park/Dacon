import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from autogluon.core.metrics import make_scorer

def IC50_to_pIC50(ic50_nM): 
    return 9 - np.log10(ic50_nM)

def pIC50_to_IC50(pIC50): 
    return 10**(9 - pIC50)

def competition_score_func(y_true_pic50: np.ndarray, y_pred_pic50: np.ndarray) -> float:
    """
    try3에서 정의된 대회 점수 계산 방식
    Score = 0.4 * A + 0.6 * B
    A: 1 - min(NRMSE_IC50, 1) (IC50 기준 정규화된 RMSE)
    B: R² score (pIC50 기준)
    """
    # IC50 값으로 변환
    y_true_ic50 = pIC50_to_IC50(y_true_pic50)
    y_pred_ic50 = pIC50_to_IC50(y_pred_pic50)
    
    # A: IC50 기준 Normalized RMSE
    rmse = np.sqrt(mean_squared_error(y_true_ic50, y_pred_ic50))
    y_range = y_true_ic50.max() - y_true_ic50.min()
    nrmse = rmse / y_range if y_range > 0 else 0
    A = 1 - min(nrmse, 1)
    
    # B: pIC50 기준 R² score
    B = r2_score(y_true_pic50, y_pred_pic50)
    
    # Final score (try3의 가중치 적용)
    score = 0.4 * A + 0.6 * B
    
    return score

# AutoGluon Scorer 생성
competition_scorer = make_scorer(
    name='competition_score',
    score_func=competition_score_func,
    optimum=1.0,  # 최적값: A=1, B=1일 때 score=1
    greater_is_better=True,  # 높을수록 좋음
    needs_pred=True  # regression 메트릭
) 