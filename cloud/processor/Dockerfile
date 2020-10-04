FROM ubuntu:latest


ENV TZ=America/Chicago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y libffi-dev python3-pip curl unzip python3-tk
RUN pip3 install -U gsutil pyyaml tqdm

RUN apt install -y vim-tiny mosquitto-clients 
RUN pip3 install paho-mqtt
RUN pip3 install boto3

RUN mkdir /usr/share/app
WORKDIR /usr/share/app
# install aws client and configure on build (hopefully)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install
# after build, on first run, need to ssh into container and execute 'aws configure' to add credentials

