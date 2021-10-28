#!/bin/bash

# select record or use the first input argument of this script
rec=$1

printf "Cleaning directories...\n"
# clean dirs
rm /media/fast_storage/matthieu_scherpf/tmp/orig/original/*
rm /media/fast_storage/matthieu_scherpf/tmp/orig/resized/*
rm /media/fast_storage/matthieu_scherpf/tmp/orig/face_detection_info/*

printf "Copying files in temporary space...\n"
# copy files
cp /home/matthieu_scherpf/repositories/GitLab/deepperfusion_ippg/data/fast_tmp_storage/2018_12_UBFC_Dataset/measurements/$rec/vid.avi /media/fast_storage/matthieu_scherpf/tmp/orig/original
python3 prepare_images_UBFC.py

printf "Running deepLab on files in temporary space...\n"
# run deeplab
docker run \
    -u $(id -u):$(id -g) \
    -v /media/fast_storage/matthieu_scherpf/tmp:/app/shared/data \
    -v /home/matthieu_scherpf/repositories/GitHub/ippg-toolbox/ippgtoolbox/detection/skin/deeplab:/app/shared/deeplab \
    -p 5000:80 \
    -p 0.0.0.0:6006:6006 \
    --gpus all \
    --rm \
    --name deepLab_segmentation \
    -d \
    -it tf_gpu:latest \
    bash dockerexec.sh

printf "Waiting for docker container to finish...\n"

docker wait deepLab_segmentation