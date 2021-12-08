from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import sys

sys.path.append('./deeplab/datasets')
import build_data
from six.moves import range
import tensorflow as tf

from pathlib import Path
import numpy as np
import cv2
from deeplab import common
import sys
import shutil
import progressbar
import io

def process_dataset(record_path, output_res):

    original_files = []
    for root, dirs, files in os.walk(record_path):
        for name in files:
            original_files.append(os.path.join(root, name))
    
    if not len(original_files):
        NotImplementedError('There were no images to be read in the specified record path.')
                
    if original_files[0].endswith('.png'):
        img_type = '.png'
    elif original_files[0].endswith('.jpg'):
        img_type = '.jpg'
    else:
        NotImplementedError('This image type can\'t be handled! Please use .jpg or .png')

    original_files = sorted(original_files)
    
    print('>>> Rescaling frames to specified shape...')
    
    progress = progressbar.ProgressBar(
                    max_value=len(original_files)-1,
                    term_width=80,
                    widget_kwargs={'marker':'▒'})
    
    image_list = [] # list with all image names
    for i, image_path in progress(enumerate(original_files)):
        dst_fname = image_path.split('/')[-1].replace(img_type, '.png')
        # transform original files to the input size for deeplab
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(
            img, dsize=output_res, interpolation=cv2.INTER_LINEAR)
        if th is not None:
            img = img >= th * 255
            img = img*255
        cv2.imwrite(image_path, img.astype(np.uint8))

    print('Finished rescaling.')

def none_or_float(value):
    if value == 'None':
        return None
    return float(value) # if value is a string (representing a float)

if __name__ == "__main__":
    segm_path='/app/shared/segmentation'
    output_res=tuple([int(x) for x in sys.argv[1].split('x')])
    th=none_or_float(sys.argv[2])

    process_dataset(segm_path, output_res)