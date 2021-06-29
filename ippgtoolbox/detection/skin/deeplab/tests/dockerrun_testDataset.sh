#!/bin/bash

# select record or use the first input argument of this script
rec=$1

printf "Cleaning directories...\n"
# clean dirs
# NOTE problems when using rm or cp because of "Argument list too long" exception -> therefore using find
rm /home/matthieu_scherpf/repositories/GitHub/ippg-toolbox/ippgtoolbox/detection/skin/deeplab/tests/assets/tmp/orig/original/*
rm /home/matthieu_scherpf/repositories/GitHub/ippg-toolbox/ippgtoolbox/detection/skin/deeplab/tests/assets/tmp/orig/resized/*
rm /home/matthieu_scherpf/repositories/GitHub/ippg-toolbox/ippgtoolbox/detection/skin/deeplab/tests/assets/tmp/orig/face_detection_info/*

printf "Copying files in temporary space...\n"
# copy files
cp /home/matthieu_scherpf/repositories/GitHub/ippg-toolbox/ippgtoolbox/detection/skin/deeplab/tests/assets/testDataset/$rec/*.jpg /home/matthieu_scherpf/repositories/GitHub/ippg-toolbox/ippgtoolbox/detection/skin/deeplab/tests/assets/tmp/orig/original

python prepare_images_testDataset.py

printf "Running deepLab on files in temporary space...\n"
# run deeplab
docker run \
    -u $(id -u):$(id -g) \
    -v /home/matthieu_scherpf/repositories/GitHub/ippg-toolbox/ippgtoolbox/detection/skin/deeplab/tests/assets/tmp:/app/shared/data \
    -v /home/matthieu_scherpf/repositories/GitHub/ippg-toolbox/ippgtoolbox/detection/skin/deeplab:/app/shared/deeplab \
    -p 5000:80 \
    -p 0.0.0.0:6006:6006 \
    --gpus all \
    --rm \
    --name deepLab_segmentation \
    -d \
    -it ippg-toolbox-deeplab:latest \
    bash dockerexec.sh

printf "Waiting for docker container to finish...\n"

docker wait deepLab_segmentation