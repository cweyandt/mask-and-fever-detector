# Accessing CSI video devices within a container:

 CLI from jetson, not within container:  

`gst-launch-1.0 nvarguscamerasrc sensor_mode=0 ! 'video/x-raw(memory:NVMM),width=3820, height=2464, framerate=21/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw,width=960, height=616' ! nvvidconv ! nvegltransform ! nveglglessink -e`
 
Enabling access from within a container:
`sudo docker run --net=host --runtime nvidia --rm --ipc=host -v /tmp/.X11-unix/:/tmp/.X11-unix/ -v /tmp/argus_socket:/tmp/argus_socket --cap-add SYS_PTRACE -e DISPLAY=$DISPLAY -it nvcr.io/nvidia/l4t-base:r32.2.1`



# Links
 - <https://www.hackster.io/pjdecarlo/custom-object-detection-with-csi-ir-camera-on-nvidia-jetson-c6d315>
 - <https://www.jetsonhacks.com/2019/04/02/jetson-nano-raspberry-pi-camera/>
 
