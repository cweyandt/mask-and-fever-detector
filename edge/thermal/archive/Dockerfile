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
  mosquitto-clients \
  cmake \
  apt-utils \
  libusb-1.0-0-dev

RUN pip3 install --no-cache --upgrade pip setuptools

# Copy and install requirements first to leverage build cache
COPY requirements.txt .

# install the other python packages
RUN pip3 install -r requirements.txt

# Install GroupGets version of libuvc that supports FLIR Lepton3
RUN git clone https://github.com/groupgets/libuvc
RUN cd libuvc && mkdir build && cd build && cmake .. && make && make install

#CMD python3 thermal/uvc-radiometry.py
