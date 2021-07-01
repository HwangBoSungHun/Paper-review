# Anomaly detection
## 1. Deep SVDD
Ruff, L., Vandermeulen, R., Goernitz, N., Deecke, L., Siddiqui, S. A., Binder, A., ... & Kloft, M. (2018, July). Deep one-class classification. In International conference on machine learning (pp. 4393-4402). PMLR.
### 개요
- 정상 데이터를 중심이 c인 Hypersphere에 가까워지도록 데이터를 Mapping &#8594; 비정상 데이터는 Hypersphere 외부에 Mapping됨  
  <img src = "./img/DeepSVDD/intro.PNG" width="80%"></center>  

### The Deep SVDD Objective  
#### 1. soft-boundary Deep SVDD objective
<img src = "./img/DeepSVDD/soft-boundary_objective.PNG" width="80%"></center> 
#### 2. One-Class Deep SVDD objective
<img src = "./img/DeepSVDD/one_class_objective.PNG" width="80%"></center> 

### Anomaly score
<img src = "./img/DeepSVDD/anomaly_score.PNG" width="80%"></center> 
