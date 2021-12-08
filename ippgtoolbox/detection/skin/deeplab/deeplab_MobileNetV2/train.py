# %%
"""
The file defines the training process.

@Author: Yang Lu
@Github: https://github.com/luyanger1799
@Project: https://github.com/luyanger1799/amazing-semantic-segmentation

"""
from utils.data_generator import ImageDataGenerator
from utils.helpers import get_dataset_info, check_related_path
from utils.callbacks import LearningRateScheduler
from utils.optimizers import *
from utils.losses import *
from utils.learning_rate import *
from utils.metrics import MeanIoU
from utils import utils
from builders import builder
import tensorflow as tf
import argparse
import os
import matplotlib.pyplot as plt

## PARAMETERS ##
model = 'DeepLabV3Plus'
base_model = 'MobileNetV2'
dataset = '/app/shared/dataset'

# training parameters
loss = 'ce'                 # The loss function for training, choices=['ce', 'focal_loss', 'miou_loss', 'self_balanced_focal_loss']
num_classes = 2             # The number of classes to be segmented, type=int
batch_size = 10             # The training batch size, type=int
valid_batch_size = 1        # The validation batch size, type=int
num_valid_images = 1059     # The number of images used for validation, type=int
num_epochs = 60             # The number of epochs to train for, type=int
initial_epoch = 0           # The initial epoch of training, type=int
steps_per_epoch = None      # The training steps of each epoch, type=int
lr_scheduler = 'poly_decay' # The strategy to schedule learning rate, choices=['step_decay', 'poly_decay', 'cosine_decay']
lr_warmup = False           # Whether to use lr warm up, type=bool
learning_rate = 7e-3        # The initial learning rate, type=float
optimizer = 'sgdw'          # The optimizer for training, choices=['sgd', 'adam', 'nadam', 'adamw', 'nadamw', 'sgdw']

checkpoint_freq = 2         # How often to save a checkpoint, type=int
validation_freq = 1         # How often to perform validation, type=int
weights = None              # The path of weights to be loaded, type=str

# augmentation parameters
random_crop = False         # Whether to randomly crop the image, type=str2bool
crop_height = 512           # The height to crop the image, type=int
crop_width = 512            # The width to crop the image, type=int
h_flip = False              # Whether to randomly flip the image horizontally, type=str2bool
v_flip = True               # Whether to randomly flip the image vertically, type=str2bool
brightness = None           # Randomly change the brightness (list), type=float, nargs='+'
rotation = 0.               # The angle to randomly rotate the image, type=float
zoom_range = 0.             # The times for zooming the image, type=float, nargs='+'
channel_shift = 0.          # The channel shift range, type=float
data_aug_rate = 0.5         # The rate of data augmentation, type=float
data_shuffle = True         # Whether to shuffle the data, type=str2
random_seed = None          # The random shuffle seed, type=int


# check related paths
paths = check_related_path(os.getcwd())

# get image and label file names for training and validation
train_image_names, train_label_names, valid_image_names, valid_label_names, _, _ = get_dataset_info(dataset)

# build the model
#net, base_model = builder(num_classes, (crop_height, crop_width), model, base_model)
from model import Deeplabv3
net = Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=num_classes, backbone='mobilenetv2', OS=8, alpha=1., activation='sigmoid')

# summary
net.summary()

# load weights
if weights is not None:
    print('Loading the weights...')
    net.load_weights(weights)

# chose loss
losses = {'ce': categorical_crossentropy_with_logits,
          'focal_loss': focal_loss(),
          'miou_loss': miou_loss(num_classes=num_classes),
          'self_balanced_focal_loss': self_balanced_focal_loss()}
loss = losses[loss] if loss is not None else categorical_crossentropy_with_logits

# chose optimizer
total_iterations = len(train_image_names) * num_epochs // batch_size
wd_dict = utils.get_weight_decays(net)
ordered_values = []
weight_decays = utils.fill_dict_in_order(wd_dict, ordered_values)

optimizers = {'adam': tf.keras.optimizers.Adam(learning_rate=learning_rate),
              'nadam': tf.keras.optimizers.Nadam(learning_rate=learning_rate),
              'sgd': tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.99),
              'adamw': AdamW(learning_rate=learning_rate, batch_size=batch_size,
                             total_iterations=total_iterations),
              'nadamw': NadamW(learning_rate=learning_rate, batch_size=batch_size,
                               total_iterations=total_iterations),
              'sgdw': SGDW(learning_rate=learning_rate, momentum=0.9, batch_size=batch_size,
                           total_iterations=total_iterations)}

# lr schedule strategy
if lr_warmup and num_epochs - 5 <= 0:
    raise ValueError('num_epochs must be larger than 5 if lr warm up is used.')

lr_decays = {'step_decay': step_decay(learning_rate, num_epochs - 5 if lr_warmup else num_epochs,
                                      warmup=lr_warmup),
             'poly_decay': poly_decay(learning_rate, num_epochs - 5 if lr_warmup else num_epochs,
                                      warmup=lr_warmup),
             'cosine_decay': cosine_decay(num_epochs - 5 if lr_warmup else num_epochs,
                                          learning_rate, warmup=lr_warmup)}
lr_decay = lr_decays[lr_scheduler]

# training and validation steps
steps_per_epoch = len(train_image_names) // batch_size if not steps_per_epoch else steps_per_epoch
validation_steps = num_valid_images // valid_batch_size

# compile the model
net.compile(optimizer="Adam",
            loss=loss,
            metrics=[MeanIoU(num_classes)])
# data generator
# data augmentation setting
train_gen = ImageDataGenerator(random_crop=random_crop,
                               rotation_range=rotation,
                               brightness_range=brightness,
                               zoom_range=zoom_range,
                               channel_shift_range=channel_shift,
                               horizontal_flip=v_flip,
                               vertical_flip=v_flip)

valid_gen = ImageDataGenerator()

train_generator = train_gen.flow(images_list=train_image_names,
                                 labels_list=train_label_names,
                                 num_classes=num_classes,
                                 batch_size=batch_size,
                                 target_size=(crop_height, crop_width),
                                 shuffle=data_shuffle,
                                 seed=random_seed,
                                 data_aug_rate=data_aug_rate)

valid_generator = valid_gen.flow(images_list=valid_image_names,
                                 labels_list=valid_label_names,
                                 num_classes=num_classes,
                                 batch_size=valid_batch_size,
                                 target_size=(crop_height, crop_width))

# callbacks setting
# checkpoint setting
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(paths['checkpoints_path'],
                          '{model}_based_on_{base}_'.format(model=model, base=base_model) +
                          'miou_{val_mean_io_u:04f}_' + 'ep_{epoch:02d}.h5'),
    save_best_only=True, period=checkpoint_freq, monitor='val_mean_io_u', mode='max')
# tensorboard setting
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=paths['logs_path'])
# learning rate scheduler setting
learning_rate_scheduler = LearningRateScheduler(lr_decay, learning_rate, lr_warmup, steps_per_epoch,
                                                verbose=1)

callbacks = [model_checkpoint, tensorboard, learning_rate_scheduler]

# begin training
print("\n***** Begin training *****")
print("Dataset -->", dataset)
print("Num Images -->", len(train_image_names))
print("Model -->", model)
print("Base Model -->", base_model)
print("Crop Height -->", crop_height)
print("Crop Width -->", crop_width)
print("Num Epochs -->", num_epochs)
print("Initial Epoch -->", initial_epoch)
print("Batch Size -->", batch_size)
print("Num Classes -->", num_classes)

print("Data Augmentation:")
print("\tData Augmentation Rate -->", data_aug_rate)
print("\tVertical Flip -->", v_flip)
print("\tHorizontal Flip -->", h_flip)
print("\tBrightness Alteration -->", brightness)
print("\tRotation -->", rotation)
print("\tZoom -->", zoom_range)
print("\tChannel Shift -->", channel_shift)

print("")

# training...
history = net.fit(train_generator,
                  steps_per_epoch=steps_per_epoch,
                  epochs=num_epochs,
                  callbacks=callbacks,
                  validation_data=valid_generator,
                  validation_steps=validation_steps,
                  validation_freq=validation_freq,
                  max_queue_size=10,
                  workers=os.cpu_count(),
                  use_multiprocessing=False,
                  initial_epoch=initial_epoch)

print(history.history.keys())

# summarize history for accuracy
miou = history.history['mean_io_u']
val_miou = history.history['val_mean_io_u']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(miou))

plt.plot(epochs, miou, 'co', label='Training mIoU')
plt.plot(epochs, val_miou, 'c', label='Validation mIoU')
plt.title('Training and validation mIoU')
plt.legend()
plt.savefig('/app/shared/Amazing-Semantic-Segmentation/plots/mIoU_DMN2.jpg')
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('/app/shared/Amazing-Semantic-Segmentation/plots/loss_DMN2.jpg')
plt.show()

# save weights
net.save(filepath=os.path.join(
    paths['weights_path'], '{model}_based_on_{base_model}.h5'.format(model=model, base_model=base_model)))
