FROM alpine:latest

# Create an alpine linux - based message forwarder container:
# docker run --name forwarder --network hw03 -ti alpine sh

# install mosquitto-clients
RUN apk update && apk add mosquitto-clients

# Install python/pip
ENV PYTHONUNBUFFERED=1
RUN apk add --update --no-cache python3 && ln -sf python3 /usr/bin/python
RUN python3 -m ensurepip
RUN pip3 install --no-cache --upgrade pip setuptools
RUN pip3 install paho-mqtt
# CMD mosquitto_sub -h mosquitto -t test_topic
# the above should block until some messages arrive
# Press Control-P Control-Q to disconnect from the container

WORKDIR = /edge/thermal
#CMD python3 fowarder.py
