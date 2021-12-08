"""
The file defines the predict process of a single RGB image.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
import os
# suppress tensorflow logging messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL

import tensorflow as tf
from model import Deeplabv3
from utils.helpers import check_related_path, get_colored_info, color_encode
from utils.utils import load_image, decode_one_hot
from tensorflow.keras.applications import imagenet_utils
from builders import builder
from PIL import Image
import numpy as np
import argparse
import sys
import cv2
import time
import progressbar


def none_or_float(value):
    if value == 'None':
        return None
    return float(value) # if value is a string (representing a float)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_dir', help='Absolute path to the source directory containing images to process.', type=str, required=True)
    parser.add_argument(
        '--dst_dir', help='Absolute path to the destination directory where the segmentation results should be saved.', type=str, required=True)
    parser.add_argument(
        '--resol', help='Resolution for the segmented frames (should be the resolution of the original input frame) in WIDTHxHEIGHT', type=str, required=True)
    parser.add_argument(
        '--th', help='Threshold that is applied to the network output if masks should directly be created (value between [0., 1.]).', type=none_or_float, required=False, default=None)

    args = parser.parse_args()

    # build the model
    # net, base_model = builder(args.num_classes, (args.crop_height, args.crop_width), args.model, args.base_model)

    net = Deeplabv3(weights=None, input_tensor=None, input_shape=(512, 512, 3),
                    classes=2, backbone='mobilenetv2', OS=8, alpha=1., activation='sigmoid')

    net.load_weights('./weights/DMN2_OS8.h5')

    # parse the resolution for resizing
    output_res = tuple([int(x) for x in args.resol.split('x')])

    # load_images
    image_names = []
    for f in os.listdir(args.src_dir):
        if f.endswith('.png') or f.endswith('.jpg'):
            image_names.append(os.path.join(args.src_dir, f))
    image_names.sort()

    progress = progressbar.ProgressBar(
        max_value=len(image_names),
        term_width=80,
        widget_kwargs={'marker': '▒'})

    for i, name in progress(enumerate(image_names)):
        img = load_image(name)
        image = cv2.resize(img, dsize=(512, 512))
        image = imagenet_utils.preprocess_input(image.astype(
            np.float32), data_format='channels_last', mode='tf')

        # add batch dimension for network input
        image = np.expand_dims(image, axis=0)

        # get the prediction
        prediction = net.predict(image, verbose=0)
        prediction = np.squeeze(prediction)

        # only take the first layer as it is a two-class-classification task
        prediction = prediction[:, :, 0]

        # resize to original size
        prediction = cv2.resize(
            prediction, output_res, interpolation=cv2.INTER_NEAREST)

        # max-min normalization
        max_val = np.max(prediction)
        min_val = np.min(prediction)
        prediction = (prediction - min_val)/(max_val - min_val)

        # apply threshold to create masks if wanted
        if args.th is not None:
            prediction = prediction >= args.th

        # boost to uint8 amplitude range [0, 255]
        prediction = prediction*255

        # change pixel data type to correct format
        prediction = prediction.astype(np.uint8)

        # get PIL file
        prediction = Image.fromarray(prediction)

        # save the prediction
        _, file_name = os.path.split(name)
        file_name = file_name.split('.')[0]
        abs_file_path = os.path.join(args.dst_dir, file_name + '_segm.png')
        prediction.save(abs_file_path, 'PNG')
