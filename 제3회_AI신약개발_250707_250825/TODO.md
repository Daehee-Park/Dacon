# AI 신약개발 경진대회 - 단계별 실험 요약

## 데이터 분석 핵심
ChEMBL(저활성, 중복/노이즈 심각)과 PubChem(고활성, 중복 적음) 두 소스가 섞여 있다. Train-Test는 완전히 분리되어 있고, 라벨 노이즈가 심각하다. 중복/노이즈를 제거하면 실제 학습 데이터가 절반 이하로 줄어든다. Test 분포는 불명확해서 domain shift 문제가 구조적으로 존재한다.

---

### Baseline
Morgan Fingerprint(2048)만 feature로 사용, RandomForest 기본 설정, 단순 5-fold CV.  
CV 0.4431±0.017, Public 0.3569. CV-Public gap -19%.  
가장 단순한 구조에서 domain shift와 feature 한계가 바로 드러난다.

---

### Try #1: Feature 확장 + Stratified CV
MorganFP에 18개 물리화학적 분자특성(MW, LogP, HBD 등) 추가, RobustScaler로 정규화, IQR 기반 이상치 제거.  
Stratified KFold(5-fold, pIC50+MW 구간)로 데이터 불균형 보정, RandomizedSearchCV로 RandomForest 하이퍼파라미터 튜닝.  
CV 0.5573±0.024, Public 0.3313. gap -41%.  
Feature 확장과 정교한 CV로 내부 성능은 올랐지만, 과적합과 domain shift가 심해진다.

---

### Try #2: Feature Selection + 앙상블
Mutual Information 기반 feature selection(800개), ExtraTrees, CatBoost, LightGBM, XGBoost 등 tree 계열 모델 8-fold Stratified CV.  
Optuna로 각 모델 하이퍼파라미터 100회 이상 탐색, 성능 우수 모델 가중 평균 앙상블.  
CV 0.5724±0.000, Public 0.3323. gap -42%.  
차원 축소와 앙상블로 CV는 개선됐지만, test domain 일반화는 여전히 실패.

---

### Try #3: Domain Adaptation 시도
Test와 유사한 샘플(MW≥450, LogP≤3.8)에 2배 가중치, MACCS keys(166) 및 VSA_EState, Chi, Kappa 등 advanced feature 추가.  
Group K-Fold(소스별) CV 시도했으나 데이터 불균형으로 Stratified로 fallback, XGBoost 구조 재설계, CatBoost 버전 호환성 해결.  
CV 0.5713±0.035, Public 0.3165. gap -45%.  
Domain adaptation, feature 확장 등 다양한 시도에도 domain shift 극복은 실패.

---

### Try #4: 단순화 + Domain-aware CV
모든 물리화학적 특성 제거, MorganFP(1024)만 사용, CatBoost 단일 모델.  
Leave-One-Source-Out(LOSO) CV로 소스 간 일반화 성능 직접 측정, Optuna로 강한 L2 정규화와 Early Stopping 적용.  
CV 0.5394±0.083, Public 미제출. ChEMBL→PubChem, PubChem→ChEMBL 간 성능 차이 큼.  
단순화와 domain-aware CV로 과적합은 줄었지만, 소스 간 일반화 한계가 명확하다.

---

### Try #5: Adversarial Feature 제거
Train/Test 구분자(LGBMClassifier)로 domain 특화 feature 식별, 상위 feature 제거 후 나머지+MorganFP로 CatBoost 단일 모델 재학습.  
Stratified 8-fold CV, Optuna로 하이퍼파라미터 튜닝.  
CV 0.5599±0.042, Public 0.2973, gap -55%.  
Domain 특화 feature 제거만으로는 domain shift 극복이 불가능.

---

### Try #6: 단순화 + 5모델 앙상블
MorganFP(2048)+핵심 분자특성 5개(MW, LogP, HBD, HBA, TPSA), StandardScaler.  
RandomForest, ExtraTrees, LightGBM, XGBoost, CatBoost 등 5개 모델 Optuna(50회) 최적화, 5-fold CV, 상위 3개 모델 가중 앙상블.  
CV 0.5757±0.020, Public 0.3125, gap -46%.  
단순화+앙상블로 내부 성능은 최고치, gap은 여전히 크다.

---

### Try #7: 데이터 품질 개선 + 소스별 분리 학습
중복/노이즈 제거(1,960→1,031), 신뢰도 등급화, CV>0.5/극단값 제거 등 데이터 품질 대폭 개선.  
ChEMBL/PubChem 분리해 CatBoost/ExtraTrees로 각각 학습, Source classifier로 Test 분포 추정, 소스별 예측 가중 앙상블.  
PubChem CatBoost CV 0.8604±0.024, ChEMBL CatBoost 0.7786±0.043, LOSO 0.6651±0.147, Public 미제출.  
노이즈 제거와 소스별 분리 학습으로 CV가 폭발적으로 향상, PubChem 모델이 특히 우수. Test 분포 추정의 불확실성은 여전.

---

## 최종 인사이트
Domain shift와 라벨 노이즈가 성능 저하의 근본 원인임을 반복적으로 확인.  
Feature engineering, 앙상블, domain adaptation 등 기술적 접근만으로는 한계가 명확하다.  
데이터 품질 개선과 소스별 분리 학습이 가장 큰 성능 향상을 가져온다.  
Test 분포 불명, CV-Public gap은 구조적 한계로 남는다.

---