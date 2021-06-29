#!/bin/bash

# read record folders
SRC=/home/matthieu_scherpf/repositories/GitHub/ippg-toolbox/ippgtoolbox/detection/skin/deeplab/tests/assets/testDataset
DST=/home/matthieu_scherpf/repositories/GitHub/ippg-toolbox/ippgtoolbox/detection/skin/deeplab/tests/assets/testDatasetDestination
TMP=/home/matthieu_scherpf/repositories/GitHub/ippg-toolbox/ippgtoolbox/detection/skin/deeplab/tests/assets/tmp/segmented
INFO=/home/matthieu_scherpf/repositories/GitHub/ippg-toolbox/ippgtoolbox/detection/skin/deeplab/tests/assets/tmp/orig/face_detection_info

RECORS=$(ls $SRC)

COUNTER=$(ls $SRC | wc -l)

find . -name "*.gitkeep" -type f -delete

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
        time bash dockerrun_testDataset.sh $f >$DST/$f/process.log
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