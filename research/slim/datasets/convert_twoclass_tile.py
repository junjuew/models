r"""Converts Munich data to TFRecords of TF-Example protos.
"""

from __future__ import absolute_import, division, print_function

import cPickle as pickle
import cv2
import json
import math
import os
import random
import sys

import fire
import tensorflow as tf
from datasets import dataset_utils

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5

# output file prefix
_OUTPUT_FILE_PREFIX = 'twoclass'

# split names
_SPLIT_NAMES = ['train', 'validation', 'test']

_META_FILE_NAME_SUFFIX = 'twoclass.meta.json'


class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(
            self._decode_jpeg, feed_dict={
                self._decode_jpeg_data: image_data
            })
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dataset_dir, mode):
    """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of pkl files. Mode represents which pkl file to use. Each pickle file should contain a dict with class name as keys and a list of image ids as values.

  Returns:
    A list of image ids, and the list of class names.
  """
    mode_pkl_file = os.path.join(dataset_dir, '{}.pkl'.format(mode))
    assert os.path.exists(mode_pkl_file)

    samples = pickle.load(open(mode_pkl_file))
    image_id_to_label = {}
    for class_name, image_list in samples.iteritems():
        for image_id in image_list:
            image_id_to_label[image_id] = class_name
    return image_id_to_label


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (_OUTPUT_FILE_PREFIX,
                                                       split_name, shard_id,
                                                       _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _get_tile(image_id, dataset_dir, tile_width, tile_height):
    image_dir = os.path.join(dataset_dir, 'photos', os.path.dirname(image_id))
    assert os.path.exists(image_dir)
    image_id = os.path.basename(image_id)

    image_id_contents = image_id.split('_')
    if len(image_id_contents) == 5:
        # tf.logging.info(
        #     ('I guessed you are using Stanford dataset. '
        #      'If not, please double-check')
        # )
        video_id = '_'.join([image_id_contents[0], image_id_contents[1]])
        (frame_id, grid_x,
         grid_y) = (image_id_contents[2], image_id_contents[3],
                    image_id_contents[4])
    elif len(image_id_contents) == 4:
        # tf.logging.info(
        #     ('I guessed you are using Okutama dataset. '
        #      'If not, please double-check')
        # )
        video_id, frame_id, grid_x, grid_y = (image_id_contents[0],
                                              image_id_contents[1],
                                              image_id_contents[2],
                                              image_id_contents[3])
    else:
        raise ValueError(
            'Not recognized image_id {} from annotations.'.format(image_id))
    frame_id = int(frame_id)
    grid_x = int(grid_x)
    grid_y = int(grid_y)
    base_image_path = os.path.join(image_dir, video_id, '{:010d}'.format(
        int(frame_id + 1))) + '.jpg'
    im = cv2.imread(base_image_path)
    if im is None:
        raise ValueError('Failed to load image: '.format(base_image_path))
    tile_x = grid_x * tile_width
    tile_y = grid_y * tile_height
    current_tile = im[tile_y:tile_y + tile_height, tile_x:tile_x + tile_width]
    ret, encoded_tile = cv2.imencode('.jpg', current_tile)
    if not ret:
        raise ValueError('Failed to encode tile: '.format(image_id))
    return encoded_tile.tobytes()


def _convert_dataset(split_name, image_ids, image_id_to_label,
                     class_names_to_ids, dataset_dir, tile_width, tile_height):
    """Converts the given image_ids to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    image_ids: A list of absolute paths to jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
    assert split_name in _SPLIT_NAMES

    num_per_shard = int(math.ceil(len(image_ids) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(
                        output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard,
                                  len(image_ids))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write(
                            '\r>> Converting image %d/%d shard %d' %
                            (i + 1, len(image_ids), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = _get_tile(image_ids[i], dataset_dir,
                                               tile_width, tile_height)
                        height, width = image_reader.read_image_dims(
                            sess, image_data)

                        class_name = image_id_to_label[image_ids[i]]
                        class_id = class_names_to_ids[class_name]

                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation', 'test']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(dataset_dir, split_name,
                                                    shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def run(dataset_dir, mode, tile_width, tile_height, validation_percentage=0.1):
    """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
    mode: either 'train' or 'test. 'train' data will be split into 
           'train' and 'validation'
    validation_percentage: when mode is 'train', the percentage that should
          go into validation set
  """
    MODES = ['train', 'test']
    if mode not in MODES:
        raise ValueError(
            'mode {} not recognized. It should be one of {}'.format(
                mode, MODES))

    if mode == 'train':
        if (validation_percentage < 0) or (validation_percentage > 1):
            raise ValueError(
                'Invalid validation_percentage {}. It should be a number between 0 to 1'.
                format(mode, MODES))

    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    if _dataset_exists(dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    image_id_to_label = _get_filenames_and_classes(dataset_dir, mode)
    class_names = sorted(list(set(image_id_to_label.values())))
    image_ids = image_id_to_label.keys()
    tf.logging.info('classes: {}'.format(class_names))
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # Divide into train and validation:
    random.seed(_RANDOM_SEED)
    random.shuffle(image_ids)

    meta_data = {}
    if mode == 'train':
        print('create train files')
        _num_validation = int(len(image_ids) * validation_percentage)
        train_ids = image_ids[_num_validation:]
        validation_ids = image_ids[:_num_validation]
        _convert_dataset('train', train_ids, image_id_to_label,
                         class_names_to_ids, dataset_dir, tile_width,
                         tile_height)
        _convert_dataset('validation', validation_ids, image_id_to_label,
                         class_names_to_ids, dataset_dir, tile_width,
                         tile_height)
        meta_data['train'] = len(train_ids)
        meta_data['validation'] = len(validation_ids)
    elif mode == 'test':
        print('create test files')
        test_ids = image_ids[:]
        _convert_dataset('test', test_ids, image_id_to_label,
                         class_names_to_ids, dataset_dir, tile_width,
                         tile_height)
        meta_data['test'] = len(test_ids)

    for split_name, value in meta_data.items():
        # record meta data file, unique to twoclass dataset
        meta_file_path = os.path.join(dataset_dir, '{}.{}'.format(
            split_name, _META_FILE_NAME_SUFFIX))
        with open(meta_file_path, 'w') as f:
            json.dump({split_name: value}, f)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    print('\nFinished converting the munich dataset!')


if __name__ == '__main__':
    fire.Fire({'run': run})
