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

RECORS=$(ls $src_dir)

COUNTER=$(ls $src_dir | wc -l)

printf "Records to process: $COUNTER...\n"

for f in $RECORS; do
    printf "###################################\n"
    printf ">>> Processing record $f...\n"
    printf "###################################\n"

    # check if record has already been processed
    if [ ! -d $dst_dir/$f ]; then
        mkdir -p $dst_dir/$f
        time bash run_deeplab_on_single_record.sh -s $src_dir/$f -d $dst_dir/$f -r $resol -t $th
    else
        echo "The record has already been processed!\n"
    fi

    printf "###################################\n"
    printf ">>> Finished processing record $f...\n"
    printf "###################################\n"
    COUNTER=$((COUNTER - 1))
    printf "Records to process: $COUNTER...\n"
done
