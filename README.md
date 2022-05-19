# Realsense_SSD_Model
Classification classes of Miniature and Finding Depth by Using Tensorflow SSD Model and Realsense D455

## 졸업작품전시회 (Graduation Product Exhibition)
![image](https://github.com/FW2022/Realsense_SSD_Model/blob/main/ImgforRM/FROMPROM.png)

숭실대학교 글로벌미디어학부 졸업전시회, FROMPROM 2022의 출품작입니다.


## SSD Model

SSD는 YOLO와 같은 1-Stage Object Detection이다. YOLO에서는 이미지를 GRID로 나누어서 각 영역에 대해 Bounding Box를 예측했다면, SSD는 CNN pyramidal feature hierarchy를 이용해 예측합니다. SSD의 첫 논문은 2016년 ECCV에서 발표됐다.

![image](https://github.com/FW2022/Realsense_SSD_Model/blob/main/ImgforRM/SSDModel.png)

이미지에서 보여지는 object들의 크기는 매우 다양하고 convolution layer 들을 통해 추출한 한 feature map만 갖고 detect 하기에는 부족하다라는 생각에서 나온 논문입니다. SSD에서는 image feature를 다양한 위치의 layer들에서 추출하여 detector와 classifier를 적용합니다. 앞쪽 layer에서는 receptive field size가 작으니 더 작은 영역을 detect 할 수 있고, 뒤쪽 layer로 갈수록 receptive field size가 커지므로 더 큰 영역을 detect 할 수 있다는 특징을 이용했습니다.

SSD는 YOLO 보다 속도나 정확도 측면에서 더 높은 성능을 보였고, 이어서 DSSD, RetinaNet 등의 논문이 발표되었습니다.
