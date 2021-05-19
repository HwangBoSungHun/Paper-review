# Detection & Segmentation
## 1. Cascade R-CNN
Cai, Z., & Vasconcelos, N. (2019). Cascade R-CNN: high quality object detection and instance segmentation. IEEE transactions on pattern analysis and machine intelligence.  
### 개요
- Multi-stage extension of the R-CNN  
  <img src = "./img/CascadeRCNN/intro.png" width="60%"></center>  
- 일반적으로 threshold IoU를 낮게 설정하면 Detector의 BBox 예측이 부정확하고(Figure 1. (a)) 높게 설정하면 BBox의 예측은 정확해지지만 Recall이 감소(객체를 적게 검출)해서 AP 감소  
- Regressor의 Output IoU가 Input IoU보다 낫기 때문에(Figure 1. (C)) Cascade R-CNN의 각 stage는 __이전 stage의 output을 사용하여 순차적으로 학습(_boostrapping과 유사함_)__  
- 이후의 stage는 이전보다 threshold IoU를 증가시킴 &#8594; stage가 진행될수록 좀 더 정확한 proposal로 학습하게 됨  

### Architecture  
  <img src = "./img/CascadeRCNN/architecture.png" width="100%"></center>  
- Figure 3. (d)에 해당하는 그림이 Cascade R-CNN의 구조
- (b)는 head가 H1으로 동일(공유)하지만 (d)는 H1, H2, H3로 세 개의 head가 학습됨  
- Figure 1. (c)에서 확인한 것처럼 Input IoU보다 Output IoU가 높기 때문에 B1보다는 B2의 IoU가 더 높기 때문에 학습이 순차적으로 진행될수록(B1 &#8594; B2 &#8594; B3) 성능 향상  

## 2. Mask R-CNN
He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask r-cnn. In Proceedings of the IEEE international conference on computer vision (pp. 2961-2969).  

## 3. YOLO
- __One-stage detector__: YOLO는 객체의 위치와 클래스를 한 단계로 파악하기 때문에 굉장히 빠르고 단순한 구조  
### 3.1. YOLO v1
Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You only look once: Unified, real-time object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 779-788).  

### 개요  
### Architecture  
### Training
- Pretrain the first 20 convolutional layers on the ImageNet classification dataset.  
- Convert the model to perform detection.  
  - Train the network for about 135 epochs on PASCAL VOC 2007 and 2012  
  - Use dropout and data augmentation to avoid overfitting  
### Training - Total loss  
#### (1) Localization loss
- Errors between the __predicted boundary box__ and __ground truth__  
#### (2) Confidence loss  
- Objectness of the box  
#### (3) Classification loss  
### Non-maximal suppression (NMS)  
- YOLO can make duplicate detections for the same object  
- To fix this, YOLO applies non-maximal suppression to remove duplications with lower confidence 
1. Select the box with the highest score among the given boxes.  
2. Calculate the IOU between the selected box and the rest of the box and remove it, if it is above threshold.  
3. Repeat the above process until the number of specific boxes remains or until there are no more boxes to choose from.  
### Limitation 
When object detection performance is __poor__  
- Multiple objects.  
- Small objects(such as flocks of birds).  
- Different bounding box ratio from the training data.  
- High-definition features.  
### Experiments 
- Real-Time  
- Less than Real-Time  

### 3.2. YOLO v2
Redmon, J., & Farhadi, A. (2017). YOLO9000: better, faster, stronger. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7263-7271).  
- Problems of YOLO v1  
  - Significant number of localization errors  
  - Relatively low recall compared to region proposal-based methods  
- The goal of YOLO v2: To improving recall and localization while maintaining classification accuracy and accurate detector that is still fast.  
### High Resolution Classifier  
- YOLO v1  
  1) In YOLO v1, we learn classifier about 224 * 224 images from beginning to end layer, and then detect 448 * 448 images.  
- YOLO v2  
  1) YOLO v2 first learned 448 * 448 images (ImageNet) and Darknet-19 classification networks (works well with high resolution images) for 10 epochs.  
  2) After learning the classification network, remove the last convolution layer of Darknet-19, Avgpool, Softmax and add 4 object detection layers(Started boundary box and object detection learning)  
  3) YOLO v2 fine tune the resulting network on detection.  
### Convolutional With Anchor Boxes  
- YOLO v1 predicts arbitrary boundary boxes. &#8594; In the real-life domain, the boundary boxes are not arbitrary.
- YOLO v2 predicts offsets to each of the anchor boxes(prior).

### 3.3. YOLO v3
Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv preprint arXiv:1804.02767.  
### 3.4. YOLO v4
Bochkovskiy, A., Wang, C. Y., & Liao, H. Y. M. (2020). Yolov4: Optimal speed and accuracy of object detection. arXiv preprint arXiv:2004.10934.  
