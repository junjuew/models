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

import glob
import itertools
import math
import numpy as np
import os
import sys

import tensorflow as tf
import pickle

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                                'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_string(
    'input_dir', None, 'Test image input dir')

tf.app.flags.DEFINE_integer(
    'label', None, 'If specified, a vector y filled with the integer label will be'
                   'added to output_file too.'
)

tf.app.flags.DEFINE_string(
    'output_file', None, "Name of output pickle file {'X': X, 'tags', tags'} "
)

FLAGS = tf.app.flags.FLAGS


def main(_):
    if not FLAGS.input_dir:
        raise ValueError('You must supply the dataset directory with --input_dir')

    if not FLAGS.output_file:
        raise ValueError('You must supply the output file with --output_file')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():

        ####################
        # Select the model #
        ####################
        # XXX set num_classes=None to get the pre-logit layers
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=None,
            is_training=False)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

        eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        def fixed_image_size_preprocessing_fn(image):
            return image_preprocessing_fn(image, eval_image_size, eval_image_size)

        ##############################################################
        # Load input images #
        ##############################################################

        batch_size = FLAGS.batch_size
        input_file_names = tf.placeholder(tf.string, shape=[None],
                                          name='input_file_names')

        # Reads an image from a file
        # decodes it into a dense tensor, and resizes it
        # to a fixed shape.
        def _parse_function(filename):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image_resized = fixed_image_size_preprocessing_fn(image_decoded)
            return image_resized

        dataset = tf.contrib.data.Dataset.from_tensor_slices(input_file_names)
        dataset = dataset.map(_parse_function)
        batched_dataset = dataset.batch(batch_size)
        iterator = batched_dataset.make_initializable_iterator()
        next_batch = iterator.get_next(name='infer_input')

        ####################
        # Define the model #
        ####################
        # logits, _ = network_fn(next_batch)
        pre_logit, _ = network_fn(next_batch)
        tf.logging.info("Pre-logit layer: " + str(pre_logit))

        pre_logit = tf.squeeze(pre_logit)

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Inferring using model from %s' % checkpoint_path)

        file_name_list = glob.glob(os.path.join(FLAGS.input_dir, '*'))

        file_name_iter = itertools.imap(
            None, *([iter(file_name_list)] * batch_size))
        last_batch_size = len(file_name_list) % batch_size

        # r_server = redis.StrictRedis(host='localhost', port=6379, db=0)
        results = []
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            tf.logging.info('model restored')

            for file_names in file_name_iter:
                sess.run(iterator.initializer, feed_dict={input_file_names:
                                                              file_names})
                outputs = sess.run(pre_logit, feed_dict={input_file_names:
                                                             file_names})

                image_ids = [os.path.splitext(os.path.basename(file_name))[0] for
                             file_name in file_names]
                outputs = outputs.tolist()

                results.extend(zip(image_ids, outputs))

            tf.logging.info('evaluating last batch')
            # evaluate last batch
            file_names = file_name_list[-last_batch_size:]
            sess.run(iterator.initializer, feed_dict={input_file_names:
                                                          file_names})
            outputs = sess.run(pre_logit, feed_dict={input_file_names:
                                                         file_names})

            image_ids = [os.path.splitext(os.path.basename(file_name))[0] for
                         file_name in file_names]
            outputs = outputs.tolist()

            results.extend(zip(image_ids, outputs))

    tf.logging.info("%d results in total" % len(results))
    tags = [r[0] for r in results]
    X = np.stack([r[1] for r in results], axis=0)
    tf.logging.info("Writing X shape =" + str(X.shape))
    out_dct = {'X': X, 'tags': tags}
    if FLAGS.label is not None:
        y = np.full(X.shape[0], FLAGS.label, dtype=np.int)
        tf.logging.info("Writing y shape =" + str(y.shape))
        out_dct['y'] = y
    tf.logging.info("Wirting to file " + FLAGS.output_file)
    pickle.dump(out_dct, open(FLAGS.output_file, 'wb'))


if __name__ == '__main__':
    tf.app.run()