#!/bin/bash
# read flags and define vars
while getopts s:d:r:p:k: flag
do
    case "${flag}" in
        # absolute path within conatiner
        s) src_dir=${OPTARG};;
        # absolute path within conatiner
        d) dst_dir=${OPTARG};;
        # Auflösung der Originalbilder in BreitexHöhe
        r) resol=${OPTARG};;
        # Ausgabe von Wahrscheinlichkeiten oder MaskenTrue/False
        p) output_probas=${OPTARG};;
        k) keep_orig_seg_results=${OPTARG};; # True/False
    esac
done

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
    --name deepLab_segmentation \
    -it ippg-toolbox-deeplab:latest \
    bash dockerexec.sh -r $resol -p $output_probas

printf "Waiting for docker container to finish...\n"

# docker wait deepLab_segmentation

printf "Docker container stopped.\n"

if [ $keep_orig_seg_results == "False" ]
then
    printf "Deleting original segmenation results. Only resized results will be kept...\n"
    rm -r $dst_dir/segmentation_results
fi