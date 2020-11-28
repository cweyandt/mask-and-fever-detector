# Thermal Container 
## Dual streaming video capture with FLIR Lepton 3 and USB Webcam

The goal of this application is to demonstrate how to stream two video threads at the same time in separate processes so that it is possible to match frames between the two cameras. In this case, the Lepton3 infrared camera has a much slower frame rate than the standard USB camera feed so this is an attempt to match up frames of two side-by-side cameras for image registration.

The [FLIR Leptop 3](https://lepton.flir.com/) camera module is connected via USB using the [PureThermal2](https://groupgets.com/manufacturers/getlab/products/purethermal-2-flir-lepton-smart-i-o-module) I/O board from [GroupGets](https://groupgets.com). The creator of this I/O board supplies several github repositories with libraries for interfacing the device. While the Purethermal board is designed to stream video over standard UVC pipelines, allowing it to work like a standard USB camera, it the captured frames do not contain accurate temperature measurement data. 

Acquiring radiometric data from the Lepton3 using the PureThermal2 board is demonstrated in GroupGets's [purethermal1-uvc-capture](https://github.com/groupgets/purethermal1-uvc-capture) repo. This python implementation uses the libuvc library using ctypes, however it requires using a forked version of the libuvc library supplied by GroupGets. The installation is as follows:

```
git clone https://github.com/groupgets/libuvc
cd libuvc
mkdir build
cd build
cmake ..
make && sudo make install
```

