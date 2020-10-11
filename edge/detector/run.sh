#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

CAMERA_INDEX=${CAMERA_INDEX:-0}
docker build -t detector-test .

docker run --net=host --privileged -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e CAMERA_INDEX=$CAMERA_INDEX -e DISPLAY=$DISPLAY \
  -it --rm detector-test ./main.py maskDetector --noninteractive


exit 0

