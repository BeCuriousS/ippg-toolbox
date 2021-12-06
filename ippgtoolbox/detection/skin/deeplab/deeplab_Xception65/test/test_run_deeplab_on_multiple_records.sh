#!/bin/bash

# rm -rf $(pwd)/test/test_output_multiple_records

# bash run_deeplab_on_multiple_records.sh \
#     -s $(pwd)/test/assets/test_run_deeplab_on_multiple_records \
#     -d $(pwd)/test/test_output_multiple_records \
#     -r 500x1000 \
#     -p False \
#     -k False

rm -rf $(pwd)/test/test_output_multiple_records

bash run_deeplab_on_multiple_records.sh \
    -s $(pwd)/test/assets/test_run_deeplab_on_multiple_records \
    -d $(pwd)/test/test_output_multiple_records \
    -r 500x1000 \
    -p True \
    -k True