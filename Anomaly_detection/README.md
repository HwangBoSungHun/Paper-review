# Anomaly detection
## 1. Deep SVDD
Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S. A., Binder, A., ... & Kloft, M. (2018, July). Deep one-class classification. In International conference on machine learning (pp. 4393-4402). PMLR.
### 개요
- 정상 데이터를 중심이 c인 Hypersphere에 가까워지도록 데이터를 Mapping &#8594; 비정상 데이터는 Hypersphere 외부에 Mapping됨  
  <img src = "./img/DeepSVDD/intro.PNG" width="100%"></center>  

### The Deep SVDD Objective  
#### 1. Soft-boundary Deep SVDD objective
- 딥러닝 모델 φ(DCAE의 encoder 사용)에 의해 Mapping된 데이터가 중심 c와 가깝고(두 번째 term) 반지름 R이 최소화 되도록(첫 번째 term) 반지름 R과 딥러닝 모델의 Weight를 학습
- Hyperparameter ν ∈ (0, 1]를 통해 Boundary에 여유를 조절할 수 있음
- 학습 시 W와 R을 번갈아가면서 최적화 진행(한 변수 고정시키고 다른 변수 학습하고, ... 반복)
  <img src = "./img/DeepSVDD/soft-boundary_objective.PNG" width="60%"></center> 
#### 2. One-Class Deep SVDD objective
- Soft-boundary Deep SVDD objective의 심플한 버전
- 반지름 R에 대한 것을 없애고 Mapping된 데이터가 최대한 c와 가까워지도록 학습
  <img src = "./img/DeepSVDD/one_class_objective.PNG" width="60%"></center> 

### Anomaly score
- One-Class Deep SVDD objective로 학습 시켰을 경우 Test 데이터를 식 (5)에 넣어서 양수가 나오면 비정상, 음수가 나오면 정상으로 판단
- Soft-boundary Deep SVDD objective로 학습 시켰을 경우 식 (5)에서 R<sup>∗</sup>을 뺀 값을 기준으로 양수가 나오면 비정상, 음수가 나오면 정상으로 판단  
  <img src = "./img/DeepSVDD/anomaly_score.PNG" width="60%"></center> 
