# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the munich dataset.

The dataset scripts used to create the dataset can be found at:
tensorflow/models/research/slim/datasets/convert_munich.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'twoclass_%s_*.tfrecord'

_META_FILE_NAME_SUFFIX = 'twoclass.meta.json'

_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 1',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    """Gets a dataset tuple with instructions for reading munich.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
    meta_file_path = os.path.join(dataset_dir, '{}.{}'.format(
        split_name, _META_FILE_NAME_SUFFIX))
    if not os.path.exists(meta_file_path):
        raise ValueError(('Cannot find meta file describing split number.'
                          'No {} in {}'.format(_META_FILE_NAME_SUFFIX,
                                               dataset_dir)))
    with open(meta_file_path) as f:
        split_to_sizes = json.load(f)
        print('find meta data split sizes: {}'.format(split_to_sizes))


#  SPLITS_TO_SIZES = {'train': 3825, 'validation': 425, 'test': 4250}
    if split_name not in split_to_sizes:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
        tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label':
        tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features,
                                                      items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=split_to_sizes[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names)
