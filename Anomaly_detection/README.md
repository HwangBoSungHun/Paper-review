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

## 2. MAD-GAN(Time series)
Li, D., Chen, D., Jin, B., Shi, L., Goh, J., & Ng, S. K. (2019, September). MAD-GAN: Multivariate anomaly detection for time series data with generative adversarial networks. In International Conference on Artificial Neural Networks (pp. 703-716). Springer, Cham.  
### 개요
- LSTM-RNN을 기본 모델(generator, discriminator)로 사용하여 GAN 기반의 비지도 다변수 이상치 탐지 방법 제안 
- 전체 변수를 동시에 고려하여 변수 간의 잠재적 상호 작용 파악 
- DR-score라는 anomaly score를 사용하여 GAN에서 생성된 generator와 discriminator를 활용하여 reconstruction loss와 discrimination loss를 통해 이상치 탐지 
- SWaT, WADI 데이터 사용  
### Anomaly Detection with Generative Adversarial Training
#### 1. MAD-GAN Architecture
  <img src = "./img/MAD-GAN/architecture.PNG" width="100%"></center>  
- 왼쪽은 일반적인 GAN
- 오른쪽은 Anomaly detection
  - Reconstruction loss: 실제 데이터를 latent space로 mapping → mapping 된 데이터를 Generator에 넣어서 데이터 생성 → 생성된 것과 실제 데이터를 latent space로 mapping한 후 reconstruction error 구함
  - Discrimination loss: discriminator가 fake라고 할수록 anomaly일 가능성 높다는 것을 사용한 듯(확인 필요)

## 3. DAGMM(Time series)
Zong, B., Song, Q., Min, M. R., Cheng, W., Lumezanu, C., Cho, D., & Chen, H. (2018, February). Deep autoencoding gaussian mixture model for unsupervised anomaly detection. In International conference on learning representations.  

## 4. GDN(Time series)
Deng, A., & Hooi, B. (2021, February). Graph neural network-based anomaly detection in multivariate time series. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 5, pp. 4027-4035).  
