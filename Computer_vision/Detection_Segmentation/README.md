# Detection & Segmentation
## 1. Cascade R-CNN
Cai, Z., & Vasconcelos, N. (2019). Cascade R-CNN: high quality object detection and instance segmentation. IEEE transactions on pattern analysis and machine intelligence.  
### 개요
- Multi-stage extension of the R-CNN  
  <img src = "./img/CascadeRCNN/intro.png" width="60%"></center>  
- 일반적으로 threshold IoU를 낮게 설정하면 Detector의 BBox 예측이 부정확하고(Figure 1. (a)) 높게 설정하면 BBox의 예측은 정확해지지만 Recall이 감소(객체를 적게 검출)해서 AP 감소  
- Regressor의 Output IoU가 Input IoU보다 낫기 때문에(Figure 1. (C)) Cascade R-CNN의 각 stage는 __이전 stage의 output을 사용하여 순차적으로 학습(boostrapping과 유사함)__  
- 이후의 stage는 이전보다 threshold IoU를 증가시킴 &#8594; stage가 진행될수록 좀 더 정확한 proposal로 학습하게 됨  

### Architecture  
  <img src = "./img/CascadeRCNN/architecture.png" width="100%"></center>  
- Figure 3. (d)에 해당하는 그림이 Cascade R-CNN의 구조
- (b)는 head가 H1으로 동일(공유)하지만 (d)는 H1, H2, H3로 세 개의 head가 학습됨  
- Figure 1. (c)에서 확인한 것처럼 Input IoU보다 Output IoU가 높기 때문에 B1보다는 B2의 IoU가 더 높기 때문에 학습이 순차적으로 진행될수록(B1 &#8594; B2 &#8594; B3) 성능 향상  
## 2. Mask R-CNN
He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask r-cnn. In Proceedings of the IEEE international conference on computer vision (pp. 2961-2969).  
## 3. YOLO
### 3.1. YOLO v1
Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).  
### 3.2. YOLO v2
Redmon, J., & Farhadi, A. (2017). YOLO9000: better, faster, stronger. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7263-7271).  
### 3.3. YOLO v3
Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.  
### 3.4. YOLO v4
Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934.  
