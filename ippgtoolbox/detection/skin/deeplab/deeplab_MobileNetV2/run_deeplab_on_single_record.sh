#!/bin/bash

# read flags and define vars
while getopts s:d:r:t: flag; do
    case "${flag}" in
    # absolute path within conatiner
    s) src_dir=${OPTARG} ;;
    # absolute path within conatiner
    d) dst_dir=${OPTARG} ;;
    # Auflösung der Originalbilder in BreitexHöhe
    r) resol=${OPTARG} ;;
    # Schwellwert der auf den Netzwerkoutput angewendet werden soll
    t) th=${OPTARG} ;;
    esac
done

# check if threshold was set; if not set it to 'None' which will be parsed in the python script later on
if [ -z "$th" ]; then
    th=None
fi

mkdir -p $dst_dir

docker run \
    -u $(id -u):$(id -g) \
    -v $src_dir:/app/shared/data \
    -v $dst_dir:/app/shared/segmentation \
    -v $(pwd):/app/shared/deeplab \
    -p 5000:80 \
    -p 0.0.0.0:6006:6006 \
    --gpus all \
    --rm \
    --name deepLab_segmentation_mnv2 \
    -it ippg-toolbox-deeplab-mnv2:latest \
    python predict.py \
    --src_dir /app/shared/data \
    --dst_dir /app/shared/segmentation \
    --resol $resol \
    --th $th

printf "Waiting for docker container to finish...\n"

printf "Docker container stopped.\n"
