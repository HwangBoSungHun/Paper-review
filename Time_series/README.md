# Time series
## 1. Informer
Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021, May). Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of AAAI.
### 개요
<img src = "./img/Informer/architecture.PNG" width="100%"></center> 
- Transformer를 개선하여 Long sequence time-series forecasting (LSTF)에 적합하도록 수정한 모델
- 기존의 딥러닝 모델(예를 들어 LSTM)을 이용한 시계열 예측의 경우 Sequence가 길어질수록 정확도와 Inference speed가 떨어진다는 문제 존재
- Transformer도 LSTF에 대해 세 가지 측면에서 한계점 존재  
  - The Quadratic computation of self-attention  
    - 1개의 layer에 대해 Self-attention의 계산량은 __O(L<sup>2</sup>)__  
    - __ProbSparse self-attention__ 으로 문제 해결  
  - The memory bottleneck in stacking layers for long inputs 
    - J개의 layer에 대해 계산량은 __O(J*L<sup>2</sup>)__  
    - __Self-attention distilling operation__ 으로 문제 해결
  - The speed plunge in predicting long outputs  
    - Inference 시 Decoder에서 Dynamic decoding 방식으로 예측(k번째 예측을 다시 input으로 넣어서 k+1번째 예측)하던 것을 수정하여 one forward step으로 바꿔서 문제 해결

### Methodology
#### Efficient Self-attention Mechanism
##### ProbSparse Self-attention
- 기존 Transformer가 Self-attention 시 O(L<sup>2</sup>) 계산해야 하는 것을 줄이기 위해서 Query 중 일부를 샘플링한 후 M으로 중요한 Query만 뽑아 Self-attention을 구하는 방법
- Query 중 U = L<sub>K</sub>lnL<sub>Q</sub>개 샘플링
- 샘플링 된 Query에 대해 M bar 구함
  <img src = "./img/Informer/M.PNG" width="100%"></center> 
- M을 기준으로 u = clnL<sub>Q</sub>개만큼의 Query(Q bar) 사용하여 Self-attention 구함
  <img src = "./img/Informer/ProbSparse.PNG" width="100%"></center> 
#### Encoder: Allowing for Processing Longer Sequential Inputs under the Memory Usage Limitation
##### Self-attention Distilling
- Soft-boundary Deep SVDD objective의 심플한 버전
- 반지름 R에 대한 것을 없애고 Mapping된 데이터가 최대한 c와 가까워지도록 학습
  <img src = "./img/Informer/encoder.PNG" width="100%"></center> 

#### Decoder: Generating Long Sequential Outputs Through One Forward Procedure
- One-
