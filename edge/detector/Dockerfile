# Base image
FROM imander/opencv-tensorflow

# Create working directory
WORKDIR /usr/src/app/

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get -y install \
  python3-matplotlib \
  python3-numpy \
  python3-pyqt5 \
  python3-scipy

# Copy and install requirements first to leverage build cache
COPY requirements.txt .

# install the other python packages
RUN pip3 install -r requirements.txt

# Copy source code
COPY . .
CMD ["./main.py", "maskDetector"]
