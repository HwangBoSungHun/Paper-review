# Image Generation
## 1. SRGAN
Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., ... & Wang, Z. (2016). Photo-realistic single image super-resolution using a generative adversarial network. arXiv preprint 2016.

### 요약
- Super Resolution(SR, 화질 개선 및 이미지 사이즈 증가) 알고리즘
- 기존 SR 알고리즘은 loss를 (pixel-wise)MSE과 PSNR로 구성 &#8594; 생성된 이미지의 질감(texture) 표현에 한계(smooth하게만 표현)
- loss를 개선하여 기존 알고리즘의 한계 극복 &#8594; Perceptual loss function(Content loss + Adversarial loss)

### Method
#### (1) Architecture
- GAN  
<img src = "./img/srgan/architecture.PNG" width="50%"></center>

#### (2) Loss function
- Perceptual loss function  
<img src = "./img/srgan/loss1.PNG"></center>

- Content loss  

  이미지 자체(pixel)를 비교하던 기존 loss를 feature map을 비교하는 loss로 변경  
  
  __기존의 pixel-wise MSE loss__  
  <img src = "./img/srgan/loss2.PNG"></center>

  __수정된 loss(Content loss)__  
  <img src = "./img/srgan/loss3.jpg" width="50%"></center>

- Advrsarial loss  
<img src = "./img/srgan/loss4.PNG"></center>  


## 2. ESRGAN
Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., ... & Change Loy, C. (2018). Esrgan: Enhanced super-resolution generative adversarial networks. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops (pp. 0-0).

### 요약
- SRGAN에서 3가지 부분(Architecture, Discriminator, Perceptual loss) 개선  

### Method  
#### (1) Architecture(Generator)  
- Batch Normalization(BN) 제거 & Residual scaling 도입  
  BN은 훈련 중에 batch의 평균과 분산을 사용하여 feature를 normalize하고 테스트 중에 전체 학습 데이터의 추정된 평균과 분산을 사용. 훈련 및 테스트 데이터의 통계값이 많이 다를 때 BN 계층은 unpleasant artifact를 도입하고 일반화 능력 제한 &#8594; BN 레이어를 제거하여 일반화 능력을 향상시키고 계산 복잡성과 메모리 사용량을 줄임   
  residual을 0과 1사이의 상수를 곱하여 scaling down하는 Residual scaling를 도입하여 안정적으로 만듦
- Residual-in-Residual Dense Block(RDDB) 도입 &#8594; higher capacity & easier to train  
<img src = "./img/esrgan/architecture1.PNG" width="50%"></center>  
<img src = "./img/esrgan/architecture2.PNG" width="50%"></center>  

#### (2) Discriminator  
- Relativistic GAN(RaGAN) 사용: 기존 GAN의 Discriminator는 real인지 fake인지 판단하는 이진 분류였다면 RaGAN의 Discriminator는 한 이미지가 다른 이미지보다 더 실제 같은지를 판단 &#8594; more realistic texture details  
  __Standard Discriminator & Relativistic Discriminator__  
<img src = "./img/esrgan/discriminator.PNG" width="50%"></center>  
  __Discriminator loss__  
<img src = "./img/esrgan/discriminator_loss.PNG" width="50%"></center>  
  __Generator loss__  
<img src = "./img/esrgan/generator_loss.PNG" width="50%"></center>  
<img src = "./img/esrgan/E.PNG" width="3%"></center>는 실제 데이터(X<sub>r</sub>) 한 개에 대해 생성된 이미지(X<sub>f</sub>)는 여러 개이므로, 모든 mini-batch의 fake data에 대해 average 취함  
  SRGAN에서는 Generator loss가 생성된 이미지에 대해서만 영향을 받지만 ESRGAN에서는 실제 데이터와 생성된 데이터 모두로부터 영향 받음  
#### (3) Perceptual loss 개선  
- activation 이전의 VGG feature 사용(SRGAN에서는 activation 이후의 feature 사용) &#8594; sharper edges & more visually pleasing results  
- activation을 취한 feature는 Sparse해진다는 문제 존재(특히 Very deep network일 경우 더욱 심함) &#8594; weak supervision & inferior performance(아래 그림에서 after activation을 보면 feature가 많이 사라짐을 알 수 있음)    
<img src = "./img/esrgan/feature_map.PNG" width="60%"></center>  
  __Total loss for the Generator__  
<img src = "./img/esrgan/L_G.PNG" width="20%"></center>  
  __L<sub>1</sub> loss(Content loss)__  
<img src = "./img/esrgan/L_1.PNG" width="20%"></center>  
  L<sub>G</sub>는 Total Generator loss이며 L<sub>percep</sub>와 L<sub>G</sub><sup>Ra</sup>(Relativistic Discriminator에서의 Generator loss), L<sub>1</sub>로 이루어짐  
  L<sub>1</sub>은 content loss로 G(x<sub>i</sub>)와 ground-truth y와의 1-norm distance에 해당함  
#### (4) Network Interpolation  
- G<sub>PSNR</sub>(PSNR-oriented network) 학습 &#8594; fine-tuning해서 G<sub>GAN</sub>(GAN-based network) 학습
- 두 모델(G<sub>PSNR</sub>, G<sub>GAN</sub>)을 보간하여 G<sub>INTERP</sub> 모델 도출  
<img src = "./img/esrgan/network_interpolation.PNG" width="40%"></center>  

### 적용 결과  
<img src = "./img/esrgan/comparison.PNG" width="50%"></center>  

## 3. iSeeBetter
Chadha, A., Britto, J., & Roja, M. M. (2020). iSeeBetter: Spatio-temporal video super-resolution using recurrent generative back-projection networks. Computational Visual Media, 6(3), 307-317.  

### 요약  
- 영상에서 현재 및 인접 프레임에 대한 공간 및 시간 정보 추출 &#8594; Video Super Resolution(VSR)  
- 기존 SR 알고리즘은 단일 이미지에 대해서 SR을 적용(SISR, Sigle-Image Super Resolution) &#8594; 영상에 적용할 경우 시간적 일관성이 부족(Lack of temporal coherency)  
- 세 가지 측면에서 Novelty  
  - Generator: Recurrent back-projection networks(RBPNs) &#8594; 현재 및 인접 프레임에 대한 공간 및 시간 정보 추출  
  - Discriminator: SRGAN &#8594; 기존 알고리즘에서 볼 수 있는 artifact를 제거하면서 SR 이미지의 자연스러움(Naturality) 개선  
  - Four-fold loss function: MSE, Perceptual, adversarial, total-variation(TV) &#8594; 이미지의 미세한 세부 사항을 잘 포착  

### Method  
#### (1) Architecture  
<img src = "./img/iSeeBetter/architecture1.jpg"></center>  
- RBPN(Generator) = SISR(Single Image Super Resolution) + MISR(Multi Image Super Resolution)  
  - SISR(Fig. 3)    
    - LR<sub>t</sub>의 사이즈를 키우는 역할  
    <img src = "./img/iSeeBetter/SISR.PNG" width="70%"></center> 
  - MISR(Fig. 4)
    - 이전 시점의 이미지와의 차이를 계산하여 변화된 정도를 반영.  
    - Projection Module은 Upscale Projection과 Down Projection으로 구성  
    - Upscale Projection: 이전 시점과 현재 시점의 차이를 기반으로 encoding하여 H<sub>t-l</sub> 생성  
    - Downscale Projection: Upscale Projection의 output인 H<sub>t-l</sub>을 입력으로 받아서 decoding하여 L<sub>t-l</sub> 생성  
    <img src = "./img/iSeeBetter/MISR.PNG" width="70%"></center>
- SRGAN(Discriminator, Fig. 5)  
  - SRGAN에서 사용한 Discriminator를 사용함  
  - 일반적인 GAN에서 사용하는 Discriminator와 동일(이진 분류)  
  <img src = "./img/iSeeBetter/Discriminator.PNG" width="70%"></center>  
#### (2) Loss function  
- Generator loss  
<img src = "./img/iSeeBetter/generator_loss.PNG" width="50%"></center>  
  - MSE loss
    - Pixel wise(PSNR과 SSIM 최적화하기에 좋음), 디테일을 못 잡는다는 문제 존재(Overly-smooth)  
    <img src = "./img/iSeeBetter/mse_loss.PNG" width="50%"></center>  
  - Perceptual loss
    - SRGAN의 Perceptual loss와 동일  
    - HR과 SR에 대해 VGG-19의 중간 feature map(after activation, before maxpooling layer) 간의 차이로 구성됨  
    <img src = "./img/iSeeBetter/perceptual_loss.PNG" width="50%"></center>  
  - Adversarial loss(G 학습에 사용)  
    - D가 생성된 이미지에 대해 진짜라고 예측하게끔 G 학습 시킴  
    <img src = "./img/iSeeBetter/adversarial_loss.PNG" width="50%"></center>  
  - Total-Variation loss(TV loss)  
    - 이웃한 픽셀과의 차이로 구성됨  
    - noise를 줄이기 위한 목적  
    <img src = "./img/iSeeBetter/TV_loss.PNG" width="50%"></center>  
- Discriminator loss  
  - 실제 이미지는 예측 확률 높게, 생성된 이미지는 예측 확률 낮게 예측하도록 구성  
  <img src = "./img/iSeeBetter/discriminator_loss.PNG" width="50%"></center>  
- 전체 frame(복수의 이미지)에 대한 loss는 아래와 같이 평균을 취하여 구함(위의 loss는 단일 이미지에 대한 것)  
<img src = "./img/iSeeBetter/avg_loss.PNG" width="40%"></center>  
