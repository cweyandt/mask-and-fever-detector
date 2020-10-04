# MIDS W251 Fall 2020 HW3

FROM alpine:latest

# Install mosquitto
RUN apk update && apk add mosquitto

# Create a bridge:
# docker network create --driver bridge hw03
# Create an alpine linux - based mosquitto container:
# docker run --name mosquitto --network hw03 -p 1883:1883 -ti alpine sh

# run mosquitto
CMD /usr/sbin/mosquitto
# Press Control-P Control-Q to disconnect from the container


