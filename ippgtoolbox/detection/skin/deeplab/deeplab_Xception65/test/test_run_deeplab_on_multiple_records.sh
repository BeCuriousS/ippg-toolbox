#!/bin/bash

rm -rf $(pwd)/test/test_output_multiple_records

bash run_deeplab_on_multiple_records.sh \
    -s $(pwd)/../test/assets/test_run_deeplab_on_multiple_records \
    -d $(pwd)/test/test_output_multiple_records \
    -r 640x420 \
    -t 0.8
