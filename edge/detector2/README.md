# Introduction (소개)
This repo contains an implementation of real-time fask mask detector in tensorflow and opencv. 

![Imgur](imgs/out.gif)

It consists of two modules: (1) face detection module and (2) mask classification module. 

![Imgur](imgs/structure.png)


Face detection module can be configured to run with (1) OpenCV's cascade face detection function, (2) OpenCV's ResNet-based detection module, or (3) YoloV3-based face detection neural network.
YoloV3-based face detection module is adopted from [YoloFACE repo]https://github.com/sthanhng/yoloface]. It provides the most accurate results but it requires GPU to be able to extract faces in realtime. It can be used when processing videos containing multiple faces. Other methods runs in realtime but it is less accurate.

Mask classification neural network is implemented using the convolutional layers from MobileNetV2 as a feature extractor. Feedfoward layers are attached to perform classification. This neural network
is trained with the mask dataset from (this repo)[https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset].

Here is a link to the youtube video showing the performance of this detection module:
(link)[https://www.youtube.com/watch?v=FIdzdLvgtT0]

# How to run (사용방법)

## 1. Install dependencies
```
pip install -r requirements.txt
```
## 2. Run the realtime mask detector
```
python maskDetector.py
```

# Note
Models are stored using `git lfs(large file support)` extension. You need to use git lfs to download model files. Please read this documentation for more support. https://git-lfs.github.com/
