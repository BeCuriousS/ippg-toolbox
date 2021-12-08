#!/bin/bash
# read flags and define vars
while getopts r:t: flag; do
    case "${flag}" in
    r) resol=${OPTARG} ;; # BreitexHöhe
    # Schwellwert der auf den Netzwerkoutput angewendet werden soll
    t) th=${OPTARG} ;;
    esac
done

# add the correct interpreter path
export PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/slim

# create the tensorflow dataset
python3 preprocess_images.py

# start inferencing
bash vis_3_1.sh

# resize images to original shape
python3 postprocess_images.py $resol $th
