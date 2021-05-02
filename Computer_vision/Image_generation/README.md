# Image Generation
## 1. SRGAN
Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., ... & Wang, Z. (2016). Photo-realistic single image super-resolution using a generative adversarial network. arXiv preprint 2016.

### 요약
- Super Resolution(SR, 화질 개선 및 이미지 사이즈 증가) 알고리즘
- 기존 SR 알고리즘은 loss를 (pixel-wise)MSE과 PSNR로 구성 --> 생성된 이미지의 질감(texture) 표현에 한계(smooth하게만 표현)
- loss를 개선하여 기존 알고리즘의 한계 극복 --> Perceptual loss function(Content loss + Adversarial loss)

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
