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
"""Inference script for tiles."""

from __future__ import absolute_import, division, print_function

import functools
import glob
import itertools
import json
import numpy as np
import os
import cPickle as pickle
import time
from PIL import Image

import redis
import tensorflow as tf
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer('batch_size', 30,
                            'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string('model_name', 'mobilenet_v1',
                           'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string('input_dir', None, 'Test image input dir')
tf.app.flags.DEFINE_string('video_ids', '2.2.2,2.2.4,1.1.7', 'Test video ids')
tf.app.flags.DEFINE_integer('redis_db', -1, 'Index of Redis Database to use')
tf.app.flags.DEFINE_string(
    'output_endpoint_names', 'Predictions,AvgPool_1a',
    'Output endpoints names.'
    'Endpoints are defined by network_fn. '
    'For mobilenet, AvgPool_1a is the extracted feature layer. '
    'Predictions is the final output layer')

tf.app.flags.DEFINE_string('result_hook', 'PickleDictHook',
                           'The Hook class used to process inference results. '
                           'One of "RedisHook", "PickleHook".')

tf.app.flags.DEFINE_string('result_file', 'inference_results.p',
                           'The file to store inference result.')
tf.app.flags.DEFINE_integer('image_w', -1, 'input image width.')
tf.app.flags.DEFINE_integer('image_h', -1, 'input image height.')
tf.app.flags.DEFINE_integer('grid_w', -1, '# of tiles horizontally.')
tf.app.flags.DEFINE_integer('grid_h', -1, '# of tiles vertically.')
tf.app.flags.DEFINE_float('max_gpu_memory_fraction', 0.3,
                          'Upper bound on the fraction of gpu memory to use.')

FLAGS = tf.app.flags.FLAGS

MODEL_CONFIG = {}
MODEL_CONFIG['mobilenet_v1'] = {
    'input_height': 224,
    'input_width': 224,
    'input_mean': 128,
    'input_std': 128,
    'num_classes': 2
}


class InferResultHook(object):
    def add_results(self, image_ids, predictions):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()


class RedisHook(InferResultHook):
    def __init__(self):
        super(RedisHook, self).__init__()
        self.r_server = redis.StrictRedis(
            host='localhost', port=6379, db=FLAGS.redis_db)

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


class PickleDictHook(InferResultHook):
    def __init__(self):
        super(PickleDictHook, self).__init__()
        self.filename = FLAGS.result_file
        self.results = {}

    def add_results(self, image_ids, predictions):
        mapping = dict(zip(image_ids, predictions))
        # tf.logging.info(json.dumps(mapping, indent=4))
        self.results.update(mapping)

    def finalize(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.results, f)


def _get_tf_preprocessing_fn(network_fn):
    preprocessing_name = FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name, is_training=False)
    eval_image_size = network_fn.default_image_size

    def fixed_image_size_preprocessing_fn(image):
        return image_preprocessing_fn(image, eval_image_size, eval_image_size)

    return fixed_image_size_preprocessing_fn


def _get_latest_checkpoint_path():
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path
    return checkpoint_path


def _create_tf_inference_graph_for_pil_tile(
        input_width, input_height, input_mean, input_std, num_classes):
    input_images = tf.placeholder(
        tf.uint8, shape=[None, None, None, 3], name='input')
    float_caster = tf.cast(input_images, tf.float32)
    resized = tf.image.resize_images(float_caster, [input_height, input_width])
    normalized_images = tf.divide(
        tf.subtract(resized, [input_mean]), [input_std])

    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name, num_classes=num_classes, is_training=False)

    logits, endpoints = network_fn(normalized_images)
    endpoint_names = FLAGS.output_endpoint_names.split(',')
    output_endpoints = []
    for endpoint_name in endpoint_names:
        assert (endpoint_name in endpoints)
        output_endpoints.append(endpoints[endpoint_name])
    return input_images, output_endpoints


def _create_tf_inference_graph_for_tf_tile(input_width, input_height,
                                           input_mean, input_std, num_classes):
    input_file_names = tf.placeholder(
        tf.string, shape=[None], name='input_file_names')

    def _image_read_and_decode(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        return image_decoded

    dataset = tf.data.Dataset.from_tensor_slices(input_file_names)
    dataset = dataset.map(_image_read_and_decode)
    dataset = dataset.batch(FLAGS.batch_size)
    iterator = dataset.make_initializable_iterator()
    input_images = iterator.get_next(name='infer_input')
    float_caster = tf.cast(input_images, tf.float32)
    tiles = _tf_divide_to_tiles(float_caster)
    resized = tf.image.resize_images(tiles, [input_height, input_width])
    normalized_images = tf.divide(
        tf.subtract(resized, [input_mean]), [input_std])

    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name, num_classes=num_classes, is_training=False)

    logits, endpoints = network_fn(normalized_images)
    endpoint_names = FLAGS.output_endpoint_names.split(',')
    output_endpoints = []
    for endpoint_name in endpoint_names:
        assert (endpoint_name in endpoints)
        output_endpoints.append(endpoints[endpoint_name])
    return (iterator, input_file_names), output_endpoints


def load_image_into_numpy_array(image):
    return np.asarray(image)


def _divide_to_tiles(im, grid_w, grid_h):
    im_h, im_w, _ = im.shape
    assert (im_w % grid_w == 0), \
        "image width ({}) cannot be evenly divided to ({}) pieces".format(
            im_w, grid_w)
    assert (im_h % grid_h == 0), \
        "image height ({}) cannot be evenly divided to ({}) pieces".format(
            im_h, grid_h)

    tile_w = int(im_w / grid_w)
    tile_h = int(im_h / grid_h)

    # row majored. if tiles are divided into 2x2
    # then the sequence is (0,0), (0,1), (1,0), (1,1)
    # in which 1st index is on x-aix, 2nd index on y-axis
    tiles = np.zeros(shape=(grid_w * grid_h, tile_h, tile_w, 3))
    for h_idx in range(0, grid_w):
        for v_idx in range(0, grid_h):
            tile_x = int(h_idx * tile_w)
            tile_y = int(v_idx * tile_h)
            current_tile = im[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]
            tiles[h_idx * grid_h + v_idx] = current_tile
    return tiles


def _get_tile_ids_from_file_path(file_path, grid_w, grid_h):
    dir_path, file_name = os.path.split(file_path)
    frame_id = int(os.path.splitext(file_name)[0])
    dir_name = os.path.basename(dir_path)
    video_id = dir_name
    image_id = '{}_{}'.format(video_id, frame_id)
    tile_ids = []
    for grid_x in range(grid_w):
        for grid_y in range(grid_h):
            tile_id = image_id + '_{}_{}'.format(grid_x, grid_y)
            tile_ids.append(tile_id)
    assert len(tile_ids) == grid_w * grid_h
    return tile_ids


def _tf_divide_to_tiles(images_tensor):
    im_w, im_h = FLAGS.image_w, FLAGS.image_h
    grid_w, grid_h = FLAGS.grid_w, FLAGS.grid_h
    assert (im_w % grid_w == 0), \
        "image width ({}) cannot be evenly divided to ({}) pieces".format(
            im_w, grid_w)
    assert (im_h % grid_h == 0), \
        "image height ({}) cannot be evenly divided to ({}) pieces".format(
            im_h, grid_h)
    tile_w = int(FLAGS.image_w / FLAGS.grid_w)
    tile_h = int(FLAGS.image_h / FLAGS.grid_h)

    # row majored. if tiles are divided into 2x2
    # then the sequence is (0,0), (0,1), (1,0), (1,1)
    # in which 1st index is on x-aix, 2nd index on y-axis
    tiles = []
    for h_idx in range(0, grid_w):
        for v_idx in range(0, grid_h):
            tile_x = h_idx * tile_w
            tile_y = v_idx * tile_h
            current_tile = tf.image.crop_to_bounding_box(
                images_tensor, tile_y, tile_x, tile_h, tile_w)
            tiles.append(current_tile)
    tiles = tf.transpose(tiles, perm=[1, 0, 2, 3, 4])
    tf.logging.info('tile shape before concat: {}'.format(tiles))
    tiles = tf.reshape(tiles, [-1, tile_h, tile_w, 3])
    tf.logging.info('tile shape after reshape: {}'.format(tiles))
    return tiles


def _evaluate_batch_tf_tile(sess, file_paths, input_op,
                            output_endpoints_op, result_hook):
    (iterator, input_op) = input_op
    sess.run(
        iterator.initializer, feed_dict={
            input_op: file_paths
        })
    st = time.time()
    tile_ids = list(
        itertools.chain.from_iterable([
            _get_tile_ids_from_file_path(image_path, FLAGS.grid_w,
                                         FLAGS.grid_h)
            for image_path in file_paths
        ]))

    outputs = sess.run(
        output_endpoints_op, feed_dict={
            input_op: file_paths
        })
    # remove unnecessary dimensions
    outputs = [np.squeeze(output) for output in outputs]
    # concatenate them together, so that outputs[0] is the
    # result for the first image
    outputs = np.concatenate(outputs, axis=1).tolist()

    tf.logging.info('inference took: {} s'.format(time.time() - st))
    st = time.time()

    result_hook.add_results(tile_ids, outputs)


def _evaluate_batch_pil_tile(sess, file_paths, input_images_op,
                             output_endpoints_op, result_hook):
    st = time.time()
    tile_ids = list(
        itertools.chain.from_iterable([
            _get_tile_ids_from_file_path(image_path, FLAGS.grid_w,
                                         FLAGS.grid_h)
            for image_path in file_paths
        ]))
    images = [Image.open(image_path) for image_path in file_paths]
    images_np = [load_image_into_numpy_array(image) for image in images]
    tf.logging.info(
        'load images into numpy array took: {} s'.format(time.time() - st))
    st = time.time()

    # TODO: probably heavy copy here, inefficient
    tiles = np.asarray(
        [_divide_to_tiles_fixed_size(im=image_np) for image_np in images_np])
    (image_num, tile_per_image, tile_h, tile_w, channel_num) = tiles.shape
    assert channel_num == 3
    assert tile_per_image == FLAGS.grid_w * FLAGS.grid_h
    # make them into a batch
    tiles = np.reshape(
        tiles, (image_num * tile_per_image, tile_h, tile_w, channel_num))
    tf.logging.info('tiling took: {} s'.format(time.time() - st))
    st = time.time()

    outputs = sess.run(output_endpoints_op, feed_dict={input_images_op: tiles})
    # remove unnecessary dimensions
    outputs = [np.squeeze(output) for output in outputs]
    # concatenate them together, so that outputs[0] is the
    # result for the first image
    outputs = np.concatenate(outputs, axis=1).tolist()

    tf.logging.info('inference took: {} s'.format(time.time() - st))
    st = time.time()

    result_hook.add_results(tile_ids, outputs)


def _get_file_paths():
    if FLAGS.video_ids:
        video_ids = FLAGS.video_ids.split(',')
        file_paths_list = []
        for video_id in video_ids:
            file_paths_list.extend(
                glob.glob(os.path.join(FLAGS.input_dir, video_id, '*')))
        return file_paths_list
    else:
        return glob.glob(os.path.join(FLAGS.input_dir, '*/*'))


def main(_):
    if FLAGS.result_hook == 'RedisHook' and FLAGS.redis_db < 0:
        raise ValueError('Invalid Redis Database index: {}'.format(
            FLAGS.redis_db))
    if (FLAGS.result_hook in [
            'PickleHook', 'PickleDictHook'
    ]) and (not os.path.exists(os.path.dirname(FLAGS.result_file))):
        raise ValueError('Directory does not exist for result file: {}'.format(
            FLAGS.result_file))
    if (FLAGS.grid_w < 0) or (FLAGS.grid_h < 0):
        raise ValueError('Invalid grid_w or grid_h')
    if FLAGS.model_name not in MODEL_CONFIG:
        raise ValueError('Unsupported model: {}. Supported models: {}'.format(
            FLAGS.model_name, MODEL_CONFIG.keys()))

    globals()['_divide_to_tiles_fixed_size'] = functools.partial(
        _divide_to_tiles, grid_w=FLAGS.grid_w, grid_h=FLAGS.grid_h)

    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=FLAGS.max_gpu_memory_fraction)

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        (input_op,
         output_endpoints_op) = _create_tf_inference_graph_for_tf_tile(
             **MODEL_CONFIG[FLAGS.model_name])
        checkpoint_path = _get_latest_checkpoint_path()
        tf.logging.info('Inferring using model from %s' % checkpoint_path)

        # file input iterator
        batch_size = FLAGS.batch_size
        file_paths_list = _get_file_paths()
        inference_num = len(file_paths_list)
        file_name_iter = itertools.imap(
            None, *([iter(file_paths_list)] * batch_size))
        last_batch_size = inference_num % batch_size

        tf.logging.info("Using result hook class: " + FLAGS.result_hook)
        result_hook = globals()[FLAGS.result_hook]()

        saver = tf.train.Saver()
        finished_num = 0
        with tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options)) as sess:
            saver.restore(sess, checkpoint_path)
            tf.logging.info('model restored')
            for file_paths in file_name_iter:
                _evaluate_batch_tf_tile(sess, file_paths, input_op,
                                        output_endpoints_op, result_hook)
                finished_num += batch_size
                tf.logging.info('finished [{}/{}]'.format(
                    finished_num, inference_num))

            if last_batch_size > 0:
                tf.logging.info('evaluating last batch')
                file_paths = file_paths_list[-last_batch_size:]
                _evaluate_batch_tf_tile(sess, file_paths, input_op,
                                        output_endpoints_op, result_hook)
                finished_num += len(file_paths)
                tf.logging.info('finished [{}/{}]'.format(
                    finished_num, inference_num))

        result_hook.finalize()


if __name__ == '__main__':
    tf.app.run()
