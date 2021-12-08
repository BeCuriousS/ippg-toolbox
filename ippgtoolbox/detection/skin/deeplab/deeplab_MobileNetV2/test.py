"""
The file defines the testing process.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from utils.data_generator import ImageDataGenerator
from utils.helpers import get_dataset_info, check_related_path
from utils.losses import categorical_crossentropy_with_logits
from utils.metrics import MeanIoU, recall
from builders import builder
import tensorflow as tf
import argparse
import os


## PARAMETERS ##
model = 'DeepLabV3Plus'
base_model = 'MobileNetV2'

dataset = '/app/shared/dataset'

num_classes = 2                     # The number of classes to be segmented, type=int
crop_height = 512                   # The height to crop the image, type=int
crop_width = 512                    # The width to crop the image, type=int
batch_size = 5                     # The training batch size, type=int
weights = '/app/shared/Amazing-Semantic-Segmentation/weights/DMN2_OS8.h5'     # The path of weights to be loaded, type=str


# check related paths
paths = check_related_path(os.getcwd())

# get image and label file names for training and validation
_, _, _, _, test_image_names, test_label_names = get_dataset_info(dataset)

# build the model
#net, base_model = builder(args.num_classes, (args.crop_height, args.crop_width), args.model, args.base_model)

from model import Deeplabv3
net = Deeplabv3(weights=None, input_tensor=None, input_shape=(512, 512, 3), classes=num_classes, backbone='mobilenetv2', OS=8, alpha=1., activation='sigmoid')

# summary
net.summary()

# load weights
print('Loading the weights...')
if weights is None:
    net.load_weights(filepath=os.path.join(
        paths['weigths_path'], '{model}_based_on_{base_model}.h5'.format(model=model, base_model=base_model)))
else:
    if not os.path.exists(weights):
        raise ValueError('The weights file does not exist in \'{path}\''.format(path=weights))
    net.load_weights(weights)

# compile the model
net.compile(optimizer=tf.keras.optimizers.Adam(),
            loss=categorical_crossentropy_with_logits,
            metrics=[MeanIoU(num_classes), tf.keras.metrics.Precision(), recall])
# data generator
test_gen = ImageDataGenerator()

test_generator = test_gen.flow(images_list=test_image_names,
                               labels_list=test_label_names,
                               num_classes=num_classes,
                               batch_size=batch_size,
                               target_size=(crop_height, crop_width))

# begin testing
print("\n***** Begin testing *****")
print("Dataset -->", dataset)
print("Model -->", model)
print("Base Model -->", base_model)
print("Crop Height -->", crop_height)
print("Crop Width -->", crop_width)
print("Batch Size -->", batch_size)
print("Num Classes -->", num_classes)

print("")

# some other training parameters
steps = len(test_image_names) // batch_size

# testing
scores = net.evaluate_generator(test_generator, steps=steps, workers=os.cpu_count(), use_multiprocessing=False)

print('loss={loss:0.4f}, MeanIoU={mean_iou:0.4f}, Precision={p:0.4f}, Sensitivity={sn:0.4f}'.format(loss=scores[0], mean_iou=scores[1], p=scores[2], sn=scores[3]))
