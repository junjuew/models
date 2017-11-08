r"""Converts Munich data to TFRecords of TF-Example protos.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import json

import tensorflow as tf
import fire

from datasets import dataset_utils

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5

# output file prefix
_OUTPUT_FILE_PREFIX = 'twoclass'

# split names
_SPLIT_NAMES = ['train', 'validation', 'test']

_META_FILE_NAME = 'twoclass.meta.json'

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  flower_root = os.path.join(dataset_dir, 'photos')
  directories = []
  class_names = []
  for filename in os.listdir(flower_root):
    path = os.path.join(flower_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (_OUTPUT_FILE_PREFIX,
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in _SPLIT_NAMES

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation', 'test']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir, mode, validation_percentage=0.1):
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
    raise ValueError('mode {} not recognized. It should be one of {}'.format(mode, MODES))

  if mode == 'train':
    if (validation_percentage < 0) or (validation_percentage > 1):
      raise ValueError('Invalid validation_percentage {}. It should be a number between 0 to 1'.format(mode, MODES))
  
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)

  meta_data = {}
  if mode == 'train':
    print('create train files')
    _num_validation = int(len(photo_filenames) * validation_percentage)
    training_filenames = photo_filenames[_num_validation:]
    validation_filenames = photo_filenames[:_num_validation]
    _convert_dataset('train', training_filenames, class_names_to_ids,
                     dataset_dir)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir)
    meta_data['train'] = len(training_filenames)
    meta_data['validation'] = len(validation_filenames)
  elif mode == 'test':
    print('create test files')
    test_filenames = photo_filenames[:]
    _convert_dataset('test', test_filenames, class_names_to_ids,
                     dataset_dir)
    meta_data['test'] = len(test_filenames)

  # record meta data file, unique to twoclass dataset
  meta_file_path = os.path.join(dataset_dir, _META_FILE_NAME)
  with open(meta_file_path, 'w') as f:
    json.dump(meta_data, f)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  print('\nFinished converting the munich dataset!')


if __name__ == '__main__':
  fire.Fire({'run': run})
