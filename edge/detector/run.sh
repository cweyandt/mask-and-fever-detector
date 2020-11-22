#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

MQTT_HOST=${MQTT_HOST:-127.0.0.1}
CAMERA_INDEX=${CAMERA_INDEX:-0}
docker build -t detector-test .

docker run --net=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e MQTT_HOST=$MQTT_HOST -e CAMERA_INDEX=$CAMERA_INDEX -e DISPLAY=$DISPLAY \
  -it --rm detector-test ./main.py maskDetector --noninteractive --mqtt --display --log-level debug

exit 0
