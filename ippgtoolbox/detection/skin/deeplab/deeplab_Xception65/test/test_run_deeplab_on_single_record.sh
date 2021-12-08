#!/bin/bash

rm -rf $(pwd)/test/test_output_single_record

bash run_deeplab_on_single_record.sh \
    -s $(pwd)/../test/assets/test_run_deeplab_on_single_record \
    -d $(pwd)/test/test_output_single_record \
    -r 640x420 \
    -t 0.8
