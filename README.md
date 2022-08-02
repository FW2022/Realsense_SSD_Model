<p align="middle">
    <img width="200px;" src="https://github.com/omizha/Space4U-client/blob/master/docs/img/logo.png?raw=true"/>
</p>
<h2 align="middle">Space for U</h2>
<p align="middle">WebGL(three.js) + AI를 활용한 인터랙티브 미디어 아트</p>
<p align="middle">https://space4-u-client.vercel.app/space</p>
<p align="middle">
    WebGL </br>
    <img src="https://img.shields.io/badge/tech-ReactJS-blue" />
    <img src="https://img.shields.io/badge/tech-ThreeJS-lightgrey" />
    <img src="https://img.shields.io/badge/tech-NestJS-red" />
    <img src="https://img.shields.io/badge/tech-GraphQL-purple" />
    </br>
    </br> AI </br>
    <img src="https://img.shields.io/badge/tech-Tenserflow-orange" />
    <img src="https://img.shields.io/badge/tech-Keras-red" />
    <img src="https://img.shields.io/badge/tech-Sklearn-blue" />    
    </br>
    </br> OpenCV </br>
    <img src="https://img.shields.io/badge/tech-Librosa-purple" />
    <img src="https://img.shields.io/badge/tech-pygame-green" />
    
</p>

## Exhibition Process
<img src="https://github.com/omizha/Space4U-client/blob/master/docs/img/ExhibitionProcess1.png?raw=true">
<img src="https://github.com/omizha/Space4U-client/blob/master/docs/img/ExhibitionProcess2.png?raw=true">

# Realsense_SSD_Model
Classification classes of Miniature and Finding Depth by Using Tensorflow SSD Model and Realsense D455

## 졸업작품전시회 (Graduation Product Exhibition)
![image](https://github.com/FW2022/Realsense_SSD_Model/blob/main/ImgforRM/FROMPROM.png)

숭실대학교 글로벌미디어학부 졸업전시회, FROMPROM 2022의 출품작입니다.


## SSD Model

SSD는 YOLO와 같은 1-Stage Object Detection이다. YOLO에서는 이미지를 GRID로 나누어서 각 영역에 대해 Bounding Box를 예측했다면, SSD는 CNN pyramidal feature hierarchy를 이용해 예측합니다. SSD의 첫 논문은 2016년 ECCV에서 발표됐다.

![image](https://github.com/FW2022/Realsense_SSD_Model/blob/main/ImgforRM/SSDModel.png)

이미지에서 보여지는 object들의 크기는 매우 다양하고 convolution layer 들을 통해 추출한 한 feature map만 갖고 detect 하기에는 부족하다라는 생각에서 나온 논문입니다. SSD에서는 image feature를 다양한 위치의 layer들에서 추출하여 detector와 classifier를 적용합니다. 앞쪽 layer에서는 receptive field size가 작으니 더 작은 영역을 detect 할 수 있고, 뒤쪽 layer로 갈수록 receptive field size가 커지므로 더 큰 영역을 detect 할 수 있다는 특징을 이용했습니다. SSD는 YOLO 보다 속도나 정확도 측면에서 더 높은 성능을 보였고, 이어서 DSSD, RetinaNet 등의 논문이 발표되었습니다.

우리 작품에 쓰인 Tensorflow Model은 Tensorflow 1 Object Dectection Model zoo의 SSD MobileNet v1 coco 2017이다. 

링크 : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md

## Realsense Depth Camera

Realsense Depth Camera D455를 사용했다. Depth Camera를 통해 물체와의 거리를 인식받아 그 거리에 따라 바다와 하늘, 우주로의 배경전환을 적용했다.


## Multi Thread
일반적으로 하나의 프로세스는 하나의 스레드를 가지고 작업을 수행한다. 하지만 멀티 스레드(multi thread)란 하나의 프로세스 내에서 둘 이상의 스레드가 동시에 작업을 수행하는 것을 의미한다. 이와 비슷하게 멀티 프로세스(multi process)는 여러 개의 CPU를 사용하여 여러 프로세스를 동시에 수행하는 것을 의미한다.

멀티 스레드와 멀티 프로세스 모두 여러 흐름을 동시에 수행하다는 공통점을 가지고 있다. 멀티 프로세스는 각 프로세스가 독립적인 메모리를 가지고 별도로 실행되지만, 멀티 스레드는 각 스레드가 자신이 속한 프로세스의 메모리를 공유한다는 점이 다르다.

멀티 스레드는 각 스레드가 자신이 속한 프로세스의 메모리를 공유하므로, 시스템 자원의 낭비가 적다. 또한, 하나의 스레드가 작업을 할 때 다른 스레드가 별도의 작업을 할 수 있어 사용자와의 응답성도 좋아진다.

### 문맥 교환(context switching)
컴퓨터에서 동시에 처리할 수 있는 최대 작업 수는 CPU의 코어(core) 수와 같다. 만약 CPU의 코어 수보다 더 많은 스레드가 실행되면, 각 코어가 정해진 시간 동안 여러 작업을 번갈아가며 수행한다. 이 때 각 스레드가 서로 교체될 때 스레드 간의 문맥 교환(context switching)이 발생한다. 문맥 교환이란 현재까지의 작업 상태나 다음 작업에 필요한 각종 데이터를 저장하고 읽어오는 작업이다.
이러한 문맥 교환에 걸리는 시간이 커지면 커질수록, 멀티 스레딩의 효율은 저하된다. 오히려 많은 양의 단순한 계산은 싱글 스레드로 동작하는 것이 더 효율적일 수 있다. 따라서 많은 수의 스레드를 실행하는 것이 언제나 좋은 성능을 보이는 것은 아니라는 점을 유의해야 한다.

참초 : http://www.tcpschool.com/java/java_thread_multi

본 작품의 2단계에 쓰인 Object Detection은 One Thread에서 입력과 출력, Display가 모두 처리되는 것보다 Multi Thread에서 각각의 단계를 수행하는 것이 더 빨라서 Multi Therad를 사용했다.

## Contributors
<table>
  <tr>
      <td align="center">
    <a>
        <img src="https://avatars.githubusercontent.com/u/30133053?v=4" width="150px;" alt="" href="https://github.com/Junst"/>
        <br />
        <sub>
            <a href="https://github.com/Junst"><b>< 고준영 ></b></a>
            <br />
            <a href="https://github.com/FW2022/MusicMoodRecognition"> 음악 분류 모델 개발</a> <br />
            <a href="https://github.com/FW2022/Realsense_SSD_Model"> 미니어처 분류 모델 개발 </a>
        </sub>
    </a>
</td>
    <td align="center">
        <a href="https://github.com/omizha">
            <img src="https://avatars.githubusercontent.com/u/4525704?v=4?s=100" width="150px;" alt=""/>
            <br />
            <sub>
                <b>< 하정훈 ></b>
                <br />
                WebGL 인터랙티브 웹 개발 <br />
                인공지능 파이프라인 서버 구축
            </sub>
        </a>
    </td>
    <td align="center">
    <a href="https://github.com/your-mistletoe">
        <img src="https://avatars.githubusercontent.com/u/84714861?v=4" width="150px;" alt=""/>
        <br />
        <sub>
            <b>< 박다인 ></b>
            <br />
            퍼지 추론 시스템 개발 <br />
            전시 연출 및 굿즈 디자인
        </sub>
    </a>
</td>
</table>

