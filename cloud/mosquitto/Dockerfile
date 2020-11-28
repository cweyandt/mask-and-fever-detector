FROM alpine:latest

# Install mosquitto
RUN apk update && apk add mosquitto

# Create a bridge:
# docker network create --driver bridge cloud

# Create an alpine linux - based mosquitto container:
# docker run --name mosquitto --network cloud -p 1883:1883 -ti alpine sh

# run mosquitto
CMD /usr/sbin/mosquitto
# Press Control-P Control-Q to disconnect from the container


