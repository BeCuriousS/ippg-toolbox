#!/bin/bash

# read record folders
SRC=/media/fast_storage/matthieu_scherpf/2019_06_26_BP4D+_v0.2/measurements
DST=/media/fast_storage/matthieu_scherpf/2019_06_26_BP4D+_v0.2/processing/ROI_deepLab/sensors_2021_ms
TMP=/media/fast_storage/matthieu_scherpf/tmp/segmented
INFO=/media/fast_storage/matthieu_scherpf/tmp/orig/face_detection_info

RECORS=$(ls $SRC)

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
        time bash dockerrun_BP4D+.sh $f >$DST/$f/process.log
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