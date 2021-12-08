"""
The file defines the evaluate process on target dataset.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from sklearn.metrics import multilabel_confusion_matrix
from utils.helpers import *
from utils.utils import load_image
import numpy as np
import argparse
import sys
import cv2
import os

dataset = '/home/paula_wilhelm/50ECU'                                                   # The path of the dataset
crop_height = 512                                                                       # The height to crop the image
crop_width = 512                                                                        # The width to crop the image
predictions = '/home/paula_wilhelm/Amazing-Semantic-Segmentation/eroded_predictions'    # The path of predicted image

# check related paths
paths = check_related_path(os.getcwd())

# get image and label file names for training and validation
_, _, _, _, _, test_label_names = get_dataset_info(dataset)

# get color info
csv_file = os.path.join(dataset, 'class_dict.csv')

class_names, _ = get_colored_info(csv_file)

# get the prediction file name list
if not os.path.exists(predictions):
    raise ValueError('the path of predictions does not exit.')

prediction_names = []
for file in sorted(os.listdir(predictions)):
    prediction_names.append(os.path.join(predictions, file))

# evaluated classes
evaluated_classes = get_evaluated_classes(os.path.join(dataset, 'evaluated_classes.txt'))

num_classes = len(class_names)
class_iou = dict()
for name in evaluated_classes:
    class_iou[name] = list()

class_p = dict()
for name in evaluated_classes:
    class_p[name] = list()

class_sn = dict()
for name in evaluated_classes:
    class_sn[name] = list()

class_idx = dict(zip(class_names, range(num_classes)))

# begin evaluate
assert len(test_label_names) == len(prediction_names)

for i, (name1, name2) in enumerate(zip(test_label_names, prediction_names)):
    sys.stdout.write('\rRunning test image %d / %d' % (i + 1, len(test_label_names)))
    sys.stdout.flush()

    label = np.array(cv2.resize(load_image(name1),
                                dsize=(crop_width, crop_height), interpolation=cv2.INTER_NEAREST))
    pred = np.array(cv2.resize(load_image(name2),
                               dsize=(crop_width, crop_height), interpolation=cv2.INTER_NEAREST))

    confusion_matrix = multilabel_confusion_matrix(label.flatten(), pred.flatten(), labels=list(class_idx.values()))
    for eval_cls in evaluated_classes:
        eval_idx = class_idx[eval_cls]
        (tn, fp), (fn, tp) = confusion_matrix[eval_idx]

        if tp + fn > 0:
            class_iou[eval_cls].append(tp / (tp + fp + fn))

print('\n****************************************')
print('* The IoU of each class is as follows: *')
print('****************************************')
for eval_cls in evaluated_classes:
    class_iou[eval_cls] = np.mean(class_iou[eval_cls])
    print('{cls:}: {iou:.4f}'.format(cls=eval_cls, iou=class_iou[eval_cls]))

print('\n**********************************************')
print('* The Mean IoU of all classes is as follows: *')
print('**********************************************')
print('Mean IoU: {mean_iou:.4f}'.format(mean_iou=np.mean(list(class_iou.values()))))


# Precision

for j, (name1, name2) in enumerate(zip(test_label_names, prediction_names)):
    sys.stdout.write('\rRunning test image %d / %d' % (j + 1, len(test_label_names)))
    sys.stdout.flush()

    label = np.array(cv2.resize(load_image(name1),
                                dsize=(crop_width, crop_height), interpolation=cv2.INTER_NEAREST))
    pred = np.array(cv2.resize(load_image(name2),
                               dsize=(crop_width, crop_height), interpolation=cv2.INTER_NEAREST))

    confusion_matrix = multilabel_confusion_matrix(label.flatten(), pred.flatten(), labels=list(class_idx.values()))
    for eval_cls in evaluated_classes:
        eval_idx = class_idx[eval_cls]
        (tn, fp), (fn, tp) = confusion_matrix[eval_idx]

        if tp + fn > 0:
            class_p[eval_cls].append(tp / (tp + fp))

print('\n****************************************')
print('* The Precision of each class is as follows: *')
print('****************************************')

for eval_cls in evaluated_classes:
    class_p[eval_cls] = np.mean(class_p[eval_cls])
    print('{cls:}: {Precision:.4f}'.format(cls=eval_cls, Precision=class_p[eval_cls]))

# Sensitivity

for k, (name1, name2) in enumerate(zip(test_label_names, prediction_names)):
    sys.stdout.write('\rRunning test image %d / %d' % (k + 1, len(test_label_names)))
    sys.stdout.flush()

    label = np.array(cv2.resize(load_image(name1),
                                dsize=(crop_width, crop_height), interpolation=cv2.INTER_NEAREST))
    pred = np.array(cv2.resize(load_image(name2),
                               dsize=(crop_width, crop_height), interpolation=cv2.INTER_NEAREST))

    confusion_matrix = multilabel_confusion_matrix(label.flatten(), pred.flatten(), labels=list(class_idx.values()))
    for eval_cls in evaluated_classes:
        eval_idx = class_idx[eval_cls]
        (tn, fp), (fn, tp) = confusion_matrix[eval_idx]

        if tp + fn > 0:
            class_sn[eval_cls].append(tp / (tp + fn))

print('\n****************************************')
print('* The Sensitivity of each class is as follows: *')
print('****************************************')

for eval_cls in evaluated_classes:
    class_sn[eval_cls] = np.mean(class_sn[eval_cls])
    print('{cls:}: {Sensitivity:.4f}'.format(cls=eval_cls, Sensitivity=class_sn[eval_cls]))
