#!/bin/bash

docker run --rm -it -d \
    --name=nangs \
    --gpus=all \
    --ipc=host \
    -p 8888:8888 \
    -v ${PWD}:/workspace \
    sensioai/nangs:dev \
    jupyter notebook --NotebookApp.token=$1 --ip=0.0.0.0 --port=8888 --allow-root --no-browser