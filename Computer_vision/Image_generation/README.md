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
  
  기존의 pixel-wise MSE loss  
  <img src = "./img/srgan/loss2.PNG"></center>

  수정된 loss(Content loss)  
  <img src = "./img/srgan/loss3.jpg" width="50%"></center>

- Advrsarial loss  
<img src = "./img/srgan/loss4.PNG"></center>  


## 2. ESRGAN
Wang, X., Yu, K., Wu, S., Gu, J., Liu, Y., Dong, C., ... & Change Loy, C. (2018). Esrgan: Enhanced super-resolution generative adversarial networks. In Proceedings of the European Conference on Computer Vision (ECCV) Workshops (pp. 0-0).

### 요약
- SRGAN에서 3가지 부분 개선  
  (1) Network structure  
  - Residual-in-Residual Dense Block(RDDB) 도입 &#8594; higher capacity & easier to train  
  - Batch Normalization(BN) 제거 & Residual scaling 도입  

  (2) Discriminator  
  - Relativistic average GAN(RaGAN) 사용: 기존 GAN의 Discriminator는 real인지 fake인지 판단하는 이진 분류였다면 RaGAN의 Discriminator는 한 이미지가 다른 이미지보다 더 실제 같은지를 판단 &#8594; more realistic texture details

  (3) Perceptual loss 개선
  - activation 이전의 VGG feature 사용(SRGAN에서는 activation 이후의 feature 사용) &#8594; sharper edges & more visually pleasing results
