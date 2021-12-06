#!/bin/bash
# read flags and define vars
while getopts r:p: flag
do
    case "${flag}" in
        r) resol=${OPTARG};; # BreitexHöhe
        p) output_probas=${OPTARG};; # True/False
    esac
done

# add the correct interpreter path
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# create the tensorflow dataset
python3 preprocess_images.py

# start inferencing
bash vis_3_1.sh $output_probas

# resize images to original shape
python3 postprocess_images.py $resol $output_probas