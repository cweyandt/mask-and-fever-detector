# Base image
FROM tensorflow/tensorflow:latest

COPY Makefile .
RUN make install

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get -y install \
  libgl1 \
  libxcb-icccm4 \
  libxcb-image0 \
  libxcb-keysyms1 \
  libxcb-randr0 \
  libxcb-render-util0 \
  libxcb-shape0 \
  libxcb-xinerama0 \
  libxcb-xkb1 \
  libxkbcommon-x11-0

RUN pip3 install --no-binary opencv-python opencv-python
