#!/bin/bash

# read record folders
SRC=/home/matthieu_scherpf/repositories/GitLab/deepperfusion_ippg/data/fast_tmp_storage/2018_12_UBFC_Dataset/measurements
DST=/home/matthieu_scherpf/repositories/GitLab/deepperfusion_ippg/data/fast_tmp_storage/2018_12_UBFC_Dataset/processing/ROI_deepLab/sensors_2021_ms
TMP=/media/fast_storage/matthieu_scherpf/tmp/segmented
INFO=/media/fast_storage/matthieu_scherpf/tmp/orig/face_detection_info

RECORS=$(find $SRC -mindepth 1 -maxdepth 1 -type d -printf '%f\n')

COUNTER=$(ls $SRC | wc -l)

printf "Records to process: $COUNTER...\n"

for f in $RECORS
do
    printf "###################################\n"
    printf ">>> Processing record $f...\n"
    printf "###################################\n"

    # check if record has already been processed
    if [ ! -e $DST/$f/face_detection_info.mat ]
    then
        mkdir -p $DST/$f
        time bash dockerrun_UBFC.sh $f >$DST/$f/process.log
        cp $TMP/* $DST/$f
        cp $INFO/* $DST/$f
    else
        echo "The record has already been processed!\n"
    fi

    printf "###################################\n"
    printf ">>> Finished processing record $f...\n"
    printf "###################################\n"
    COUNTER=$((COUNTER - 1))
    printf "Records to process: $COUNTER...\n"
done