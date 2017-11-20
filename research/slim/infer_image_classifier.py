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
import redis

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer('batch_size', 100,
                            'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string('dataset_name', 'imagenet',
                           'The name of the dataset to load.')

tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string('model_name', 'inception_v3',
                           'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string('input_dir', None, 'Test image input dir')
tf.app.flags.DEFINE_integer('redis_db', -1, 'Index of Redis Database to use')
tf.app.flags.DEFINE_string(
    'output_endpoint_names', 'Predictions,AvgPool_1a',
    'Output endpoints names.'
    'Endpoints are defined by network_fn. '
    'For mobilenet, AvgPool_1a is the extracted feature layer. '
    'Predictions is the final output layer')

tf.app.flags.DEFINE_string('result_hook', 'RedisHook', 'The Hook class used to process inference results. '
                                                       'One of "RedisHook", "PickleHook".')

tf.app.flags.DEFINE_string('result_file', 'inference_results.p', 'The file to store inference result.')

FLAGS = tf.app.flags.FLAGS


class InferResultHook(object):
    def add_results(self, image_ids, predictions):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()


class RedisHook(InferResultHook):
    def __init__(self):
        super(RedisHook, self).__init__()
        self.r_server = redis.StrictRedis(host='localhost', port=6379, db=FLAGS.redis_db)

    def add_results(self, image_ids, predictions):
        mappings = dict(zip(image_ids, predictions))
        tf.logging.info(mappings)
        self.r_server.mset(mappings)

    def finalize(self):
        pass


class PickleHook(InferResultHook):
    def __init__(self):
        super(PickleHook, self).__init__()
        self.filename = FLAGS.result_file
        self.results = []

    def add_results(self, image_ids, predictions):
        l = zip(image_ids, predictions)
        tf.logging.info('\n'.join(map(str, l)))
        self.results.extend(l)

    def finalize(self):
        pickle.dump(self.results, open(self.filename, 'wb'))


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError(
            'You must supply the dataset directory with --dataset_dir')

    if FLAGS.redis_db < 0:
        raise ValueError('Invalid Redis Database index: {}'.format(FLAGS.redis_db))

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        ######################
        # Select the dataset #
        ######################
        dataset_wrapper = dataset_factory.get_dataset(
            FLAGS.dataset_name, 'train', FLAGS.dataset_dir)

        ####################
        # Select the model #
        ####################
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(
                dataset_wrapper.num_classes - FLAGS.labels_offset),
            is_training=False)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name, is_training=False)

        eval_image_size = network_fn.default_image_size

        def fixed_image_size_preprocessing_fn(image):
            return image_preprocessing_fn(image, eval_image_size,
                                          eval_image_size)

        ##############################################################
        # Load input images #
        ##############################################################

        batch_size = 100
        input_file_names = tf.placeholder(
            tf.string, shape=[None], name='input_file_names')

        # Reads an image from a file
        # decodes it into a dense tensor, and resizes it
        # to a fixed shape.
        def _parse_function(filename):
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image_resized = fixed_image_size_preprocessing_fn(
                image_decoded)
            return image_resized

        try:
            dataset = tf.data.Dataset.from_tensor_slices(input_file_names)
        except AttributeError:
            dataset = tf.contrib.data.Dataset.from_tensor_slices(input_file_names)

        dataset = dataset.map(_parse_function)
        batched_dataset = dataset.batch(batch_size)
        iterator = batched_dataset.make_initializable_iterator()
        next_batch = iterator.get_next(name='infer_input')

        ####################
        # Define the model #
        ####################
        logits, endpoints = network_fn(next_batch)
        endpoint_names = FLAGS.output_endpoint_names.split(',')
        output_endpoints = []
        for endpoint_name in endpoint_names:
            assert (endpoint_name in endpoints)
            output_endpoints.append(endpoints[endpoint_name])
            
        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(
                FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Inferring using model from %s' % checkpoint_path)

        file_name_list = glob.glob(os.path.join(FLAGS.input_dir, '*'))

        file_name_iter = itertools.imap(
            None, *([iter(file_name_list)] * batch_size))
        inference_num = len(file_name_list)
        last_batch_size = inference_num % batch_size

        tf.logging.info("Using result hook class: " + FLAGS.result_hook)
        result_hook = globals()[FLAGS.result_hook]()

        saver = tf.train.Saver()
        finished_num = 0
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            tf.logging.info('model restored')

            for file_names in file_name_iter:
                sess.run(
                    iterator.initializer,
                    feed_dict={
                        input_file_names: file_names
                    })
                outputs = sess.run(output_endpoints, feed_dict={
                    input_file_names: file_names})
                image_ids = [
                    os.path.splitext(os.path.basename(file_name))[0]
                    for file_name in file_names
                ]

                # remove unnecessary dimensions
                outputs = [np.squeeze(output) for output in outputs]
                # concatenate them together, so that outputs[0] is the
                # result for the first image
                outputs = np.concatenate(outputs, axis=1).tolist()

                result_hook.add_results(image_ids, outputs)

                finished_num += batch_size
                tf.logging.info(
                    'finished [{}/{}]'.format(finished_num, inference_num))

            if last_batch_size > 0:
                tf.logging.info('evaluating last batch')
                # evaluate last batch
                file_names = file_name_list[-last_batch_size:]
                sess.run(
                    iterator.initializer,
                    feed_dict={
                        input_file_names: file_names
                    })
                outputs = sess.run(output_endpoints, feed_dict={
                    input_file_names: file_names})
                image_ids = [
                    os.path.splitext(os.path.basename(file_name))[0]
                    for file_name in file_names
                ]

                # remove unnecessary dimensions
                outputs = [np.squeeze(output) for output in outputs]
                # concatenate them together, so that outputs[0] is the
                # result for the first image
                outputs = np.concatenate(outputs, axis=1).tolist()
                result_hook.add_results(image_ids, outputs)

                finished_num += len(mappings)
                tf.logging.info(
                    'finished [{}/{}]'.format(finished_num, inference_num))

        result_hook.finalize()


if __name__ == '__main__':
    tf.app.run()
