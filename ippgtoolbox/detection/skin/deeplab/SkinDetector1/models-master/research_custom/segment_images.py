from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os.path
import sys

sys.path.append('./deeplab/datasets')
import build_data
from six.moves import range
import tensorflow as tf

from pathlib import Path
import numpy as np
import cv2
from deeplab import common

def process_matthieu_set(txt_file_name,
                        dataset_path,
                        annotation_path,
                        txt_path):
    """
    images have to be in .jpg-format
    images have to have 512x512 resolution; if resolution is different change the resolution of the dummy-annotations, too
    images have to be located in dataset_path
    txt_file_name: name of txt-file, e.g. "dataset_to_segment.txt"
    dataset_path: path to images that you want to segment
    annotation_path: path to save dummy-annotations
    txt_path: path to save txt-file
    """
    # create missing folder if it does not exist
    Path(txt_path).mkdir(parents=True, exist_ok=True)
    Path(annotation_path).mkdir(parents=True, exist_ok=True)


    original_files = [os.path.join(root, name)
                for root, dirs, files in os.walk(dataset_path)
                for name in files
                if name.endswith(".png")]
    original_files = sorted(original_files)

    image_list = [] # list with all image names
    for image_path in original_files:
        ann = np.zeros((512,512,1)) # change if resolution is different
        cv2.imwrite(annotation_path + "/" + image_path.split('/')[-1].split('.png')[0] + '.png',ann)
        image_list.append(image_path.split('/')[-1].split('.png')[0])


    with open(txt_path + "/" + txt_file_name, 'w') as f:
        for item in image_list:
            f.write("%s\n" % item)


def convert_dataset_to_tf_format(dataset_path, annotation_path, txt_path, tf_dataset_path):
    # create missing folder if it does not exist
    Path(tf_dataset_path).mkdir(parents=True, exist_ok=True)

    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_string('image_folder',
                            dataset_path,
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
        sys.stdout.write('Processing ' + dataset)
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
                for i in range(start_idx, end_idx):
                    sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                        i + 1, len(filenames), shard_id))
                    sys.stdout.flush()
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
                    print(seg_filename)
                    seg_data = tf.gfile.GFile(seg_filename, 'rb').read()
                    seg_height, seg_width = label_reader.read_image_dims(seg_data)
                    if height != seg_height or width != seg_width:
                        raise RuntimeError('Shape mismatched between image and label.')
                    # Convert to tf example.
                    example = build_data.image_seg_to_tfexample(
                        image_data, filenames[i], height, width, seg_data)
                    tfrecord_writer.write(example.SerializeToString())
            sys.stdout.write('\n')
            sys.stdout.flush()


    def main(unused_argv):
        dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*.txt'))
        for dataset_split in dataset_splits:
            _convert_dataset(dataset_split)

    tf.app.run(main)




if __name__ == "__main__":
    txt_file_name = "DatasetToSegment.txt"
    dataset_path='/app/shared/data/orig/resized'
    annotation_path='/app/shared/data/annotations'
    txt_path='/app/shared/data/information'
    tf_dataset_path ="/app/shared/data/tf_dataset"

    process_matthieu_set(txt_file_name, dataset_path, annotation_path, txt_path)
    convert_dataset_to_tf_format(dataset_path, annotation_path, txt_path, tf_dataset_path)
