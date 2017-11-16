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

import os

import tensorflow as tf
from datasets import dataset_factory
from nets import nets_factory
from tensorflow.python.tools import freeze_graph
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer('batch_size', 1,
                            'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string('dataset_name', 'twoclass',
                           'The name of the dataset to load.')

tf.app.flags.DEFINE_string('dataset_dir', None,
                           'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string('model_name', 'mobilenet_v1',
                           'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string(
    'input_image_width', 224, 'Width of expected input images')
tf.app.flags.DEFINE_string(
    'input_image_height', 224, 'Height of expected input images')
tf.app.flags.DEFINE_string('output_node_names',
                           'MobilenetV1/Predictions/Reshape_1',
                           'The name of the output nodes, comma separated.')
tf.app.flags.DEFINE_string('output_dir', None, 'Output model path')
tf.app.flags.DEFINE_integer('placeholder_type_enum',
                            dtypes.float32.as_datatype_enum,
                            'The AttrValue enum to use for placeholders')

FLAGS = tf.app.flags.FLAGS


def optimize_frozen_graph(input_graph_file, input_names, output_names,
                          output_path):
    """Optimize frozen graph for inference. See
       tensorflow.python.tools.optimize_for_inference

    Args:
      input_graph_file: 
      input_names: 
      output_names: 
      output_path: 

    Returns:

    """
    input_graph_def = graph_pb2.GraphDef()
    with gfile.Open(input_graph_file, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        input_names.split(","),
        output_names.split(","), FLAGS.placeholder_type_enum)

    f = gfile.FastGFile(output_path, "w")
    f.write(output_graph_def.SerializeToString())


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError(
            'You must supply the dataset directory with --dataset_dir')

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

        ##############################################################
        # input images should be already processed
        ##############################################################
        input_images = tf.placeholder(
            tf.float32, shape=[None,
                               FLAGS.input_image_width,
                               FLAGS.input_image_height,
                               3], name='input')

        ####################
        # Define the model #
        ####################
        _, endpoints = network_fn(input_images)
        predictions = endpoints['Predictions']

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(
                FLAGS.checkpoint_path)
        else:
            checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Loading model from %s' % checkpoint_path)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, checkpoint_path)
            tf.logging.info('model restored')
            tf.logging.info('saving the graph without input pipeline')

            output_model_path = os.path.join(FLAGS.output_dir,
                                             'no_preprocessing_model')
            model_path = saver.save(
                sess,
                output_model_path,
                global_step=0)
            graph_path = tf.train.write_graph(sess.graph, FLAGS.output_dir,
                                              'no_preprocessing_graph.pbtxt')

    input_graph_path = graph_path
    input_saver_def_path = ""
    input_binary = False
    output_node_names = FLAGS.output_node_names
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(
        FLAGS.output_dir, 'frozen_graph.pb')
    clear_devices = True
    variable_names_blacklist = ""
    input_model_path = model_path

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,
                              input_binary, input_model_path,
                              output_node_names, restore_op_name,
                              filename_tensor_name, output_graph_path,
                              clear_devices, variable_names_blacklist)
    tf.logging.info('optimizing graph for inference')
    optimized_graph_path = os.path.join(FLAGS.output_dir,
                                        'optimized_frozen_graph.pb')
    optimize_frozen_graph(output_graph_path, 'input', output_node_names,
                          optimized_graph_path)


if __name__ == '__main__':
    tf.app.run()
