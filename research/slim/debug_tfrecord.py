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
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import, division, print_function

import cv2

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

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


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    input_file_list = [
        '/home/junjuew/mobisys18/processed_dataset/okutama/experiments/classification_896_896/twoclass_train_00000-of-00005.tfrecord'
    ]
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
            cv2.imwrite('/tmp/{}_{}.jpg'.format(i, label), im)
            # example, l = sess.run([image, label])
            # img = Image.fromarray(example, 'RGB')
            # img.save("output/" + str(i) + '-train.png')

            # print(example, l)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
