# AI 신약개발 경진대회 - 개선 로드맵

## Baseline 결과
### CV Score (5-fold)
- Average Normalized RMSE: 0.1140 ± 0.0369
- Average Correlation²: 0.1479 ± 0.0278  
- **Average Final Score: 0.4431 ± 0.0170**

### Public LB Score
- **Public Score: 0.3569403553**
- CV vs Public Gap: -0.0862 (약 19% 하락)

---

## Try #1 - 기본 모델 개선
### 우선순위 높음
- [ ] **하이퍼파라미터 튜닝**
  - RandomForest: n_estimators, max_depth, min_samples_split 최적화
  - GridSearchCV 또는 RandomizedSearchCV 사용
  
- [ ] **Feature Engineering**
  - Morgan fingerprint radius 변경 (1, 3, 4 시도)
  - 다른 molecular descriptor 추가 (MACCS keys, RDKit descriptors)
  - Fingerprint bit size 실험 (1024, 4096)

- [ ] **모델 다양화**
  - XGBoost/LightGBM 시도
  - Support Vector Regression
  - Neural Network (간단한 MLP)

### 우선순위 중간  
- [ ] **데이터 전처리 개선**
  - Outlier 제거 방법 개선
  - IC50 값 분포 분석 및 전처리
  - SMILES 정규화 (canonical SMILES)

- [ ] **Cross-validation 전략**
  - Stratified CV (pIC50 값 기준)
  - Time-based split 고려

### 우선순위 낮음
- [ ] **앙상블 방법**
  - 여러 모델 조합
  - Voting/Averaging/Stacking
  
---

## 분석 포인트
- CV-Public gap 원인 분석 필요
- Training data와 test data의 molecular diversity 차이 확인
- Overfitting 방지 전략 수립