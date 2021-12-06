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

RECORS=$(ls $src_dir)

COUNTER=$(ls $src_dir | wc -l)

printf "Records to process: $COUNTER...\n"

for f in $RECORS
do
    printf "###################################\n"
    printf ">>> Processing record $f...\n"
    printf "###################################\n"

    # check if record has already been processed
    if [ ! -d $dst_dir/$f ]
    then
        mkdir -p $dst_dir/$f
        time bash run_deeplab_on_single_record.sh -s $src_dir/$f -d $dst_dir/$f -r $resol -p $output_probas -k $keep_orig_seg_results
    else
        echo "The record has already been processed!\n"
    fi

    printf "###################################\n"
    printf ">>> Finished processing record $f...\n"
    printf "###################################\n"
    COUNTER=$((COUNTER - 1))
    printf "Records to process: $COUNTER...\n"
done