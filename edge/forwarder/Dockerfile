ARG REPO
#FROM $REPO/opencv-tensorflow
FROM imander/opencv-tensorflow

WORKDIR /edge/forwarder

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get -y install python3-numpy

# Copy and install requirements first to leverage build cache
COPY requirements.txt .

# install the python packages
RUN pip3 install -r requirements.txt

# Copy source code
COPY main.py .
CMD ["./main.py"]
