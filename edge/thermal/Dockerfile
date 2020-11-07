FROM imander/opencv-tensorflow 

# Create an alpine linux - based message forwarder container:
# docker run --name forwarder --network hw03 -ti alpine sh

# Create working directory
WORKDIR /usr/src/app/

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get -y install \
  python3-matplotlib \
  python3-numpy \
  python3-pyqt5 \
  python3-scipy \
  mosquitto-clients

RUN pip3 install --no-cache --upgrade pip setuptools

# Copy and install requirements first to leverage build cache
COPY requirements.txt .

# install the other python packages
RUN pip3 install -r requirements.txt

RUN apt-get -y install cmake apt-utils

# Install libusb requirement for libuvc
RUN apt-get -y install libusb-dev libusb-1.0-0-dev 

# Install GroupGets version of libuvc that supports FLIR Lepton3
RUN git clone https://github.com/groupgets/libuvc
RUN cd libuvc && mkdir build && cd build && cmake .. && make && make install

#WORKDIR = /edge/thermal
#CMD python3 fowarder.py
