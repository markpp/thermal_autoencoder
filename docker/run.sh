#!/bin/bash
xhost +local:
docker run -it --net=host \
  --volume=/dev:/dev \
  --name=dockerpytorch \
  --workdir=/home/code \
  -e LOCAL_USER_ID=`id -u $USER` \
  -e DISPLAY=$DISPLAY \
  -e QT_GRAPHICSSYSTEM=native \
  -e CONTAINER_NAME=dockerpytorch-dev \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  -v "$HOME/datasets:/home/datasets" \
  -v "$HOME/github/thermal_autoencoder:/home/code" \
  dockerpytorch:latest
