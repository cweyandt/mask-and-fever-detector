FROM imander/opencv-tensorflow 

# Create an alpine linux - based message forwarder container:
# docker run --name forwarder --network hw03 -ti alpine sh

# Create working directory
# WORKDIR /usr/src/app/

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get -y install \
  python3-matplotlib \
  python3-numpy \
  python3-pyqt5 \
  python3-scipy \
  mosquitto-clients

RUN pip3 install --no-cache --upgrade pip setuptools
RUN pip3 install paho-mqtt

WORKDIR = /edge/thermal
#CMD python3 fowarder.py
