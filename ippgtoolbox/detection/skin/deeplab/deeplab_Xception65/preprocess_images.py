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

def process_dataset(record_path,
                    annotation_path,
                    txt_path):
    """
    images have to be in .jpg-format
    images have to have 512x512 resolution; if resolution is different change the resolution of the dummy-annotations, too
    images have to be located in record_path
    record_path: path to images that you want to segment
    annotation_path: path to save dummy-annotations
    txt_path: path to save txt-file
    """

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
    
    print('>>> Rescaling frames for correct deeplab input...')
    
    progress = progressbar.ProgressBar(
                    max_value=len(original_files)-1,
                    term_width=80,
                    widget_kwargs={'marker':'▒'})
    
    image_list = [] # list with all image names
    for i, image_path in progress(enumerate(original_files)):
        dst_fname = image_path.split('/')[-1].replace(img_type, '.png')
        # transform original files to the input size for deeplab
        img = cv2.imread(image_path)
        img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        new_full_fname = os.path.join(resized_path, dst_fname)
        cv2.imwrite(new_full_fname, img)
        # create dummy annotation file for image
        ann = np.zeros((512,512,1))
        new_full_fname = os.path.join(annotation_path, dst_fname)
        cv2.imwrite(new_full_fname, ann)
        # save file name to list
        image_list.append(image_path.split('/')[-1].split(img_type)[0])

    print('Finished rescaling.')

    with open(txt_path + "/" + 'DatasetToSegment.txt', 'w') as f:
        for item in image_list:
            f.write("%s\n" % item)


def convert_dataset_to_tf_format(resized_path, 
                                 annotation_path, 
                                 txt_path, 
                                 tf_dataset_path):

    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_string(
        'image_folder',
        resized_path,
        'Folder containing images.')

    tf.app.flags.DEFINE_string(
        'semantic_segmentation_folder',
        annotation_path,
        'Folder containing semantic segmentation annotations.')

    tf.app.flags.DEFINE_string(
        'list_folder',
        txt_path,
        'Folder containing lists for training and validation')

    tf.app.flags.DEFINE_string(
        'output_dir',
        tf_dataset_path,
        'Path to save converted SSTable of TensorFlow examples.')


    # NOTE: if using more than one split the data gets shuffled
    _NUM_SHARDS = 1


    def _convert_dataset(dataset_split):
        """Converts the specified dataset split to TFRecord format.

        Args:
            dataset_split: The dataset split (e.g., train, test).

        Raises:
            RuntimeError: If loaded image and label have different shape.
        """
        dataset = os.path.basename(dataset_split)[:-4]
        print('>>> Converting to tensorflow dataset...')
        filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
        num_images = len(filenames)
        num_per_shard = int(math.ceil(num_images / _NUM_SHARDS))

        image_reader = build_data.ImageReader('png', channels=3)
        label_reader = build_data.ImageReader('png', channels=1)

        for shard_id in range(_NUM_SHARDS):
            output_filename = os.path.join(
                FLAGS.output_dir,
                '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                start_idx = shard_id * num_per_shard
                end_idx = min((shard_id + 1) * num_per_shard, num_images)
                
                progress = progressbar.ProgressBar(
                    max_value=end_idx-start_idx-1,
                    term_width=80,
                    widget_kwargs={'marker':'▒'})
                
                for i in progress(range(start_idx, end_idx)):
                    # sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    #     i + 1, len(filenames), shard_id))
                    # sys.stdout.flush()
                    # Read the image.
                    image_filename = os.path.join(
                        #MH:
                        #FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
                        FLAGS.image_folder, filenames[i] + '.png')
                        #END MH
                    image_data = tf.gfile.GFile(image_filename, 'rb').read()
                    height, width = image_reader.read_image_dims(image_data)
                    # Read the semantic segmentation annotation.
                    seg_filename = os.path.join(
                        FLAGS.semantic_segmentation_folder,
                        #MH:
                        #filenames[i] + '.' + FLAGS.label_format)
                        filenames[i] + '.png')
                        #END MH
                    # print(seg_filename)
                    seg_data = tf.gfile.GFile(seg_filename, 'rb').read()
                    seg_height, seg_width = label_reader.read_image_dims(seg_data)
                    if height != seg_height or width != seg_width:
                        raise RuntimeError('Shape mismatched between image and label.')
                    # Convert to tf example.
                    example = build_data.image_seg_to_tfexample(
                        image_data, filenames[i], height, width, seg_data)
                    tfrecord_writer.write(example.SerializeToString())
            # sys.stdout.write('\n')
            # sys.stdout.flush()
            print('Finished conversion.')


    def main(unused_argv):
        dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*.txt'))
        for dataset_split in dataset_splits:
            _convert_dataset(dataset_split)
            
        print('Finished frame preprocessing.')

    tf.app.run(main)




if __name__ == "__main__":
    record_path='/app/shared/data'
    
    # suppress tensorflow output
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
    
    base_path = '/home/any/processing'
    
    txt_path = os.path.join(base_path, 'information')
    tf_dataset_path = os.path.join(base_path, 'tf_dataset')
    annotation_path = os.path.join(base_path, 'annotations')
    resized_path = os.path.join(base_path, 'resized')

    # delete existing paths and files
    shutil.rmtree(base_path, ignore_errors=True)

    # create necessary paths
    Path(txt_path).mkdir(parents=True, exist_ok=True)
    Path(tf_dataset_path).mkdir(parents=True, exist_ok=True)
    Path(annotation_path).mkdir(parents=True, exist_ok=True)
    Path(resized_path).mkdir(parents=True, exist_ok=True)

    process_dataset(
        record_path, annotation_path, txt_path)
    convert_dataset_to_tf_format(
        resized_path, annotation_path, txt_path, tf_dataset_path)