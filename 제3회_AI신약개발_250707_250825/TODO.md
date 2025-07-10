# AI 신약개발 경진대회 - 개선 로드맵

## Baseline 결과
### CV Score (5-fold)
- Average Normalized RMSE: 0.1140 ± 0.0369
- Average Correlation²: 0.1479 ± 0.0278  
- **Average Final Score: 0.4431 ± 0.0170**

### Public LB Score
- **Public Score: 0.3569403553**
- CV vs Public Gap: -0.0862 (약 19% 하락)

### 구현 내용
- Morgan Fingerprint (2048 bits, radius=2) 단독 사용
- RandomForest 기본 설정
- 단순 5-fold CV
- 기본적인 데이터 전처리

---

## 데이터 분석 결과
### 주요 발견사항
- **데이터 불균형**: ChEMBL(pIC50=6.53) vs PubChem(pIC50=9.18) 활성도 차이 큼
- **Domain Shift**: Train-Test 분자 특성 차이 존재
  - Test MW 더 높음 (461.69 vs 427.52)
  - Test LogP 더 낮음 (3.59 vs 3.96)
  - Test 데이터가 더 uniform한 분포
- **pIC50 범위**: 3.30~13.00 (10배 차이, 매우 넓음)
- **Train-Test overlap**: 0% (완전히 다른 화합물)

### 문제점 진단
1. **CV-Public 갭 원인**:
   - Domain shift (분자 특성 분포 차이)
   - 데이터 불균형으로 인한 overfitting
   - Test 데이터가 train보다 더 큰 분자들

2. **모델 성능 제한 요인**:
   - 단순한 Morgan fingerprint 사용
   - 데이터소스 간 활성도 편향 미반영
   - 분자 크기/특성 차이 미고려

---

## Try #1 결과 - 도메인 지식 기반 모델 개선
### CV Score (5-fold)
- Average Normalized RMSE: 0.1107 ± 0.0426
- Average Correlation²: 0.3359 ± 0.0285
- **Average Final Score: 0.5573 ± 0.0235**

### Public LB Score
- **Public Score: 0.3313010254**
- CV vs Public Gap: -0.2260 (약 41% 하락)

### 구현 세부사항
#### Feature Engineering
- **Morgan Fingerprint**: 2048 bits, radius=2
- **분자 특성 18개 추가**: MW, LogP, HBD, HBA, TPSA, Rotatable, Aromatic_Rings, Heavy_Atoms, Lipinski_Violations, SlogP, SMR, LabuteASA, BalabanJ, BertzCT, FractionCsp3, RingCount, MolLogP, MolMR
- **정규화**: RobustScaler (이상치에 강건)
- **최종 Feature 차원**: 2066 (2048 + 18)

#### 전처리 개선
- **이상치 제거**: IQR 기반 (factor=2.0, 관대한 기준)
- **pIC50 변환**: IC50 최소값 1e-3으로 클리핑
- **데이터소스 균형**: ChEMBL vs PubChem 분포 모니터링

#### Cross-Validation 전략
- **Stratified K-Fold**: pIC50 구간 + 분자량 구간 조합
- **구간 설정**: pIC50 [0,5,7,9,15], MW 4분위
- **5-fold CV**: 데이터소스별 균형 유지

#### 모델 최적화
- **RandomizedSearchCV**: 100회 시행
- **하이퍼파라미터**: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap
- **최적화 기준**: neg_mean_squared_error

### 개선 사항
✅ **CV Score 대폭 개선**: 0.4431 → 0.5573 (+25.8%)
✅ **분자 특성 활용**: 도메인 지식 기반 물리화학적 특성
✅ **Stratified CV**: 데이터 불균형 고려한 validation
✅ **하이퍼파라미터 최적화**: 체계적인 탐색

### 문제점 발견
❌ **CV-Public 갭 확대**: -19% → -41% (overfitting 심화)
❌ **Public Score 하락**: 0.3569 → 0.3313 (-7.2%)
❌ **Feature 과적합**: 2066차원이 데이터 대비 과도
❌ **모델 복잡도**: 단일 모델로는 일반화 한계

### 교훈 및 개선 방향
- 분자 특성 추가는 효과적이나 차원수 조절 필요
- Domain shift 문제는 feature만으로 해결 불가
- 정규화 강화 및 모델 다양화 필요

---

## Try #2 결과 - Feature Selection + 앙상블 최적화
### CV Score (8-fold)
- **Ensemble: 0.5724 ± 0.0000**
- CatBoost: 0.5693 ± 0.0170
- ExtraTrees: 0.5631 ± 0.0156  
- LightGBM: 0.5599 ± 0.0163
- XGBoost: 0.4354 ± 0.0230

### Public LB Score
- **Public Score: 0.3323109265**
- CV vs Public Gap: -0.2401 (약 42% 하락)

### 구현 세부사항
#### Feature Selection 효과
- **Mutual Information 기반**: 2066 → 800개 feature 선택
- **성능 향상**: 차원 감소에도 CV 성능 개선 (0.5573 → 0.5724, +2.7%)
- **Overfitting 완화**: Feature 수 61% 감소로 일반화 개선 시도

#### Cross-Validation 개선
- **8-fold CV**: 더 안정적인 평가 (기존 5-fold 대비)
- **동일한 Stratified 전략**: pIC50 + MW 기반 계층화 유지

#### 다중 모델 앙상블 성능
- **ExtraTrees**: RandomForest 대신 사용, 빠른 속도 + 좋은 성능
- **CatBoost**: 가장 높은 개별 모델 성능 (0.5693)
- **LightGBM**: 안정적인 중간 성능 (0.5599)  
- **XGBoost**: 예상보다 낮은 성능 (0.4354) - 하이퍼파라미터 문제 의심
- **앙상블**: 개별 모델보다 우수한 성능 달성

#### Optuna 하이퍼파라미터 최적화
- **탐색 횟수**: 모델당 100회
- **최적화 목표**: CV RMSE 최소화
- **결과 저장/재사용**: 중간 실패 시에도 이미 완료된 최적화 활용
- **XGBoost 이슈**: 버전 호환성 문제로 성능 저하 가능성

### 달성된 개선사항
✅ **CV Score 추가 개선**: 0.5573 → 0.5724 (+2.7%)
✅ **Feature 효율성**: 800개 feature로 더 좋은 성능
✅ **앙상블 효과**: 개별 모델보다 안정적인 성능
✅ **8-fold CV**: 더 신뢰성 있는 평가
✅ **자동화**: Optuna + 결과 저장으로 실험 효율성 증대

### 지속되는 문제점
❌ **CV-Public 갭 악화**: -41% → -42% (미미한 악화)
❌ **Public Score 정체**: 0.3313 → 0.3323 (+0.3%, 거의 변화없음)
❌ **XGBoost 성능**: 다른 모델 대비 현저히 낮음
❌ **Domain Shift 미해결**: 여전히 근본적 문제 존재

### 기술적 문제점 발견
🐛 **LightGBM Verbose**: early_stopping verbose=False로 해결 (Callback 안의 early_stopping 내에서)
🐛 **CatBoost Early Stopping**: 버전 호환성 문제 발생 가능성
🐛 **XGBoost 호환성**: early_stopping_rounds 파라미터 이슈

### 새로운 인사이트
1. **Feature Selection 효과**: 차원 감소가 오히려 성능 향상
2. **Tree 계열 모델 우세**: ExtraTrees, CatBoost가 상위 성능
3. **XGBoost 이슈**: 하이퍼파라미터 또는 데이터 특성 부적합
4. **CV 개선 vs Public 정체**: Domain shift가 핵심 병목

---

## Try #3 - Domain Adaptation 집중 전략
### 목표
- **Domain Shift 근본 해결**: Test 분포 중심 학습 전략
- **Group-based CV**: 데이터소스 및 분자 특성 기반 validation
- **Technical Debt 해결**: 발견된 버전 호환성 문제 해결
- **Sample Weighting**: Test-like 화합물에 가중치 부여

### 우선순위 최고 (Domain Adaptation)
- [ ] **Test-like Sample Weighting**
  - MW ≥ 450 & LogP ≤ 3.8 화합물에 2배 가중치
  - Test 분포와 유사한 샘플 집중 학습
  - Sample weights를 모든 모델에 적용

- [ ] **Group-based Cross-Validation**
  - 데이터소스별 Group K-Fold (ChEMBL vs PubChem)
  - 분자량 기반 Group CV 추가
  - Leave-one-group-out validation 실험

- [ ] **XGBoost 완전 재설계**
  - 하이퍼파라미터 범위 대폭 수정
  - Regularization 강화 (alpha, lambda 범위 확장)
  - Learning rate 더 낮게, n_estimators 더 많이

### 우선순위 높음 (Technical Issues)
- [ ] **버전 호환성 완전 해결**
  - CatBoost early stopping 안전한 구현
  - XGBoost 모든 버전 호환 처리
  - LightGBM verbose 최종 검증

- [ ] **Advanced Feature Engineering**
  - MACCS keys (166 bits) 추가
  - Extended Connectivity Fingerprint (ECFP4)
  - RDKit 2D/3D descriptor 확장

### 우선순위 중간
- [ ] **앙상블 가중치 최적화**
  - CV 성능 기반 가중 평균
  - Optuna로 앙상블 가중치 최적화
  - Out-of-fold 기반 메타모델

---

## 분석 포인트
- ✅ CV-Public gap 원인: Domain shift가 핵심 병목
- ✅ Try #1: 분자 특성 추가로 CV 대폭 개선
- ✅ Try #2: Feature selection + 앙상블로 CV 추가 개선, Public 정체
- ✅ 기술적 문제점: 버전 호환성 이슈들 식별
- 🔥 **Try #3 핵심**: Domain adaptation + 기술적 완성도

---

## 실험 기록
| Try | CV Score | Public Score | CV-Public Gap | Feature 수 | 주요 변경사항 |
|-----|----------|--------------|---------------|-----------|---------------|
| Baseline | 0.4431±0.017 | 0.3569 | -19% | 2048 | Morgan fingerprint + RF |
| Try #1 | 0.5573±0.024 | 0.3313 | -41% | 2066 | +18 분자특성, Stratified CV, 하이퍼파라미터 튜닝 |
| Try #2 | 0.5724±0.000 | 0.3323 | -42% | 800 | Feature selection, ExtraTrees+3부스팅 앙상블, Optuna, 8-fold CV |
| Try #3 | TBD | TBD | TBD | TBD | Domain adaptation, Group CV, Sample weighting, 기술적 완성도 |

---

## 핵심 인사이트
1. **Feature Selection의 놀라운 효과**: 차원 감소(2066→800)에도 성능 향상
2. **앙상블의 안정성**: 개별 모델보다 일관되게 우수한 성능  
3. **CV 개선 ≠ Public 개선**: Domain shift가 가장 큰 장벽
4. **Tree 계열 모델 우세**: ExtraTrees, CatBoost가 이 데이터에 적합
5. **XGBoost 부진**: 하이퍼파라미터 또는 데이터 특성 문제 의심
6. **Domain Adaptation 필수**: CV 개선만으로는 Public 향상 한계
7. **기술적 완성도**: 버전 호환성 문제 해결이 안정성에 중요