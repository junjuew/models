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
"""Save the first 100 images from a twoclass tfrecord file to disk for
inspection."""

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import os

import tensorflow as tf


slim = tf.contrib.slim

tf.app.flags.DEFINE_string('input_file', None, 'Input TF Record file')
tf.app.flags.DEFINE_string('output_dir', None, 'Where the images are saved to')
FLAGS = tf.app.flags.FLAGS


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpg'),
            'image/class/label':
            tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        })
    return features['image/encoded'], features['image/class/label']


def create_dir_if_not_exist(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    assert FLAGS.input_file
    assert FLAGS.output_dir
    create_dir_if_not_exist(FLAGS.output_dir)

    input_file_list = [FLAGS.input_file]

    with tf.Session() as sess:
        filename_queue = tf.train.string_input_producer(input_file_list)
        encoded_image_op, label_op = read_and_decode(filename_queue)
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(100):
            encoded_image, label = sess.run([encoded_image_op, label_op])
            encoded_image = np.asarray(
                bytearray(encoded_image), dtype=np.uint8)
            im = cv2.imdecode(encoded_image, cv2.CV_LOAD_IMAGE_UNCHANGED)
            cv2.imwrite(
                os.path.join(FLAGS.output_dir, '{}_{}.jpg'.format(i, label)),
                im)
            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
