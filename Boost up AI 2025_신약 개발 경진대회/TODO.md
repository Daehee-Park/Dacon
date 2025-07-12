## 성능 비교 분석
- Baseline (Morgan FP만): Public Score 0.6989 ⭐
- try1 (복잡한 특성): Public Score 0.5992
- **교훈**: 단순함이 때로는 더 효과적

## 데이터 전처리 (RDKit 기반) - try1 완료
- [x] RDKit 설치 및 환경 확인  
- [x] SMILES 정합성 검사 후 Canonical SMILES 변환  
- [x] 염 · 용매 분리 및 표준화(Charge / Tautomer 정규화)  
- [x] 중복·이상치(ID, SMILES, Inhibition 값) 제거  
- [x] 기본 화학 특성 계산 (과도한 특성으로 인한 성능 저하 확인)
- [x] Fingerprint 생성 (Morgan만으로도 충분함 확인)
- [ ] ~~Fragment / Functional group 카운트~~ (우선순위 낮음)
- [ ] ~~3D 엔베딩 후 3D Descriptor 계산~~ (우선순위 낮음)
- [ ] ~~Mordred 또는 PaDEL 추가 descriptor~~ (우선순위 낮음)
- [x] Feature selection (과도한 특성 제거 필요)
- [x] 스케일링 및 정규화  
- [x] 학습/검증 분리(stratified KFold, seed 고정)  
- [x] 전처리 파이프라인 joblib 저장  
- [x] 전처리된 X_train/X_test CSV 및 numpy 저장  

## try2 개선 전략 (baseline 0.6989 → 0.7+ 목표)
### 1. 단순하면서 효과적인 특성 엔지니어링
- [ ] Morgan Fingerprint 파라미터 최적화
  - [ ] radius 실험 (1, 2, 3, 4)
  - [ ] nBits 실험 (1024, 2048, 4096)
  - [ ] 다양한 radius 조합 (r=2&3, r=1&2&3)
- [ ] 핵심 분자 특성만 선별 추가
  - [ ] MW, LogP, tPSA, HBD/HBA (5-10개 핵심 특성)
  - [ ] CYP3A4 관련 특성 (분자 크기, 소수성)

### 2. 외부 데이터 활용 (ChEMBL)
- [ ] ChEMBL CYP3A4 데이터 수집 및 정제
- [ ] 외부 데이터와 train 데이터 결합
- [ ] Domain adaptation 기법 적용

### 3. 타겟 변환 실험
- [ ] Log transformation: log(Inhibition + 1)
- [ ] Quantile transformation (uniform/normal)
- [ ] Power transformation (Box-Cox, Yeo-Johnson)
- [ ] 변환 후 성능 비교

### 4. 모델 최적화
- [ ] AutoGluon vs PyCaret 비교 실험
- [ ] Hyperparameter tuning with Optuna
- [ ] 앙상블 가중치 최적화
- [ ] Pseudo-labeling 기법 적용

### 5. 교차 검증 및 평가
- [ ] Stratified KFold 구현 (타겟 분포 고려)
- [ ] Group KFold (유사 분자 그룹 고려)
- [ ] 커스텀 메트릭 최적화 (0.5*RMSE + 0.5*Corr)

### 6. 후처리 및 제출 최적화
- [ ] 앙상블 예측값 후처리
- [ ] 예측값 범위 최적화 (0-100 clipping)
- [ ] 여러 시드 평균 앙상블

## 우선순위 작업 (try2)
1. **Morgan FP 파라미터 튜닝** (가장 중요)
2. **타겟 변환 실험**
3. **외부 데이터 활용**
4. **모델 앙상블 최적화**
5. **Pseudo-labeling**

## 폐기된 접근법
- ~~복잡한 다중 fingerprint 조합~~ (성능 저하 확인)
- ~~과도한 분자 특성 계산~~ (overfitting 유발)
- ~~3D descriptor 계산~~ (computational cost 대비 효과 낮음)