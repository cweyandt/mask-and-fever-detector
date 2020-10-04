FROM nvcr.io/nvidia/l4t-ml:r32.4.3-py3
#FROM ubuntu:latest

# tested on Jetson NX


# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy contents
COPY . /usr/src/app

WORKDIR /usr/src/app/

# Install dependencies (pip or conda)
#RUN pip3 install -r requirements.txt


RUN apt update && apt install -y libffi-dev python3-pip curl unzip python3-tk libopencv-dev python3-opencv 
RUN pip3 install -U gsutil pyyaml tqdm cython torchvision 
RUN apt install -y python3-scipy python3-matplotlib python3-numpy
# RUN pip3 install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
RUN pip3 install -y vim
