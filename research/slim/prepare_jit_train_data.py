import ast
import pickle
from operator import itemgetter

import fire
import glob

import itertools
import numpy as np
import redis
import os

#
# Examples usage:
# python prepare_jit_train_data.py make_tp_fp_dataset
#   --redis-host 172.17.0.10
#   --file_glob "/home/zf/opt/drone-scalable-search/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/2_more_test/tile_test_by_label/*/bookstore_video0_video*"
#   --output_file /home/zf/opt/drone-scalable-search/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/2_more_test/bookstore_video0_tp_fp.p
#
from sklearn.metrics import confusion_matrix


def _split_imageid(image_id):
    tokens = image_id.split('_')
    video_id, frame_id, grid_x, grid_y = '_'.join(tokens[:-3]), tokens[-3], tokens[-2], tokens[-1]
    frame_id = int(frame_id)
    grid_x = int(grid_x)
    grid_y = int(grid_y)
    return video_id, frame_id, grid_x, grid_y


def _combine_imageid(*args):
    return '_'.join(map(str, args))


def _increment_frame_id(image_id):
    video_id, frame_id, grid_x, grid_y = _split_imageid(image_id)
    return _combine_imageid(video_id, frame_id+1, grid_x, grid_y)


def make_tp_fp_dataset_2(tile_classification_annotation_file,
                         tile_test_inference_file,
                         output_file=None,
                         max_samples=3000,
                         ):
    """

    :param tile_classification_annotation_file: gives ground truth of a video (imageid -> True/False)
    :param tile_test_inference_file: gives inference result of a video (imageid -> list[1026])
    :param output_file:
    :param max_samples:
    :param over_sample_ratio:
    :return:
    """
    tile_ground_truth = pickle.load(open(tile_classification_annotation_file))
    print("Loaded {} results from {} ".format(len(tile_ground_truth), tile_classification_annotation_file))

    tile_inference_result_and_pre_logit = pickle.load(open(tile_test_inference_file))
    print("Loadecd {} results from {}".format(len(tile_inference_result_and_pre_logit), tile_test_inference_file))

    # XXX bring the 1-off frame ids in sync!
    tile_ground_truth = dict((_increment_frame_id(k), 1 if v else 0)
                             for k, v in tile_ground_truth.iteritems())
    tile_pre_logit = dict((k, np.array(v[2:])) for k, v in tile_inference_result_and_pre_logit.iteritems())
    tile_predictions = dict((k, np.argmax(np.array(v[:2]))) for k, v in tile_inference_result_and_pre_logit.iteritems())

    # for x in [tile_ground_truth, tile_pre_logit, tile_predictions]:
    #     print(str(x.items()[:5000:1000]))

    assert set(tile_ground_truth.keys()) == set(tile_predictions.keys())
    sorted_imageids = sorted(tile_ground_truth.keys())  # hopefully sorts by timestamps
    # print("Intersect {} imageids".format(len(sorted_imageids)))
    print("Sorted {} imageids".format(len(sorted_imageids)))
    ground_truth = np.array([tile_ground_truth[id] for id in sorted_imageids])
    predictions = np.array([tile_predictions[id] for id in sorted_imageids])
    pre_logit = np.stack([tile_pre_logit[id] for id in sorted_imageids])

    print(ground_truth.shape, predictions.shape, pre_logit.shape)

    print("Confusion matrix")
    cm = confusion_matrix(y_true=ground_truth, y_pred=predictions)
    print(str(cm))
    n_fp = cm[0, 1]
    n_tp = cm[1, 1]
    max_samples = min(max_samples, n_fp, n_tp)
    print("Sampling at most {} per TP/FP.".format(max_samples))

    fp_mask = np.logical_and(ground_truth == 0, predictions == 1)
    tp_mask = np.logical_and(ground_truth == 1, predictions == 1)
    assert np.count_nonzero(fp_mask) == n_fp
    assert np.count_nonzero(tp_mask) == n_tp

    p = float(max_samples) / n_fp
    sample_fp_mask = np.logical_and(fp_mask,
                                    np.random.choice([True, False],
                                                     size=fp_mask.shape,
                                                     p=[p, 1.0 - p]))
    p = float(max_samples) / n_tp
    sample_tp_mask = np.logical_and(tp_mask,
                                    np.random.choice([True, False],
                                                     size=tp_mask.shape,
                                                     p=[p, 1.0 - p]))

    print("Sampled {} TPs and {} FPs.".format(np.count_nonzero(sample_tp_mask),
                                              np.count_nonzero(sample_fp_mask)))

    sample_mask = np.logical_or(sample_fp_mask, sample_tp_mask)

    X_out = pre_logit[sample_mask]
    y_out = ground_truth[sample_mask]

    print X_out.shape, y_out.shape

    if output_file is not None:
        dct = {'X': X_out, 'y': y_out, 'image_ids': itemgetter(*(np.nonzero(sample_mask)[0].tolist()))(sorted_imageids)}
        pickle.dump(dct,
                    open(output_file, 'wb'))


def make_tp_fp_dataset(file_glob, redis_host, output_file=None, max_samples=3000, over_sample_ratio=20):
    """
    Separate one video's INFERRED positive tiles into TP and FP.
    :param over_sample_ratio: Over sample in order to find sufficient number of FPs,
        while avoiding loading all inference results. In rare cases, you may need to increase this number and rerun.
    :param redis_host:
    :param output_file: a pickle file {'X': np.array, 'y': np.array, 'image_ids': tuple<str>}
    :param file_glob: the wildcard giving a video's all tiles. Ground truth is guessed from the file paths.
    :param max_samples: Maximum number of TP samples to retain
    :return:
    """

    paths = glob.glob(file_glob)
    image_ids_and_ground_truth_1 = [(filename2imageid(p), 1) for p in paths if 'positive' in p]
    image_ids_and_ground_truth_0 = [(filename2imageid(p), 0) for p in paths if 'negative' in p]

    max_samples = min(max_samples, len(image_ids_and_ground_truth_1))
    # we must retain all ground truth 1 tiles because they are scarce
    # sample ground truth 0 tiles to reduce amount of data loading
    # we keep 10X sample sizes because there will be true negatives
    image_ids_and_ground_truth_0 = image_ids_and_ground_truth_0[
                                   ::int(1 + len(image_ids_and_ground_truth_0) / (over_sample_ratio * max_samples))]

    image_ids_and_ground_truth = sorted(
        image_ids_and_ground_truth_1 + image_ids_and_ground_truth_0)  # hopefully it sorts tiles based on timestamps

    print '\n'.join(map(str, image_ids_and_ground_truth[::len(image_ids_and_ground_truth) / 10]))

    image_ids = [x[0] for x in image_ids_and_ground_truth]
    ground_truth = np.array([x[1] for x in image_ids_and_ground_truth])

    assert len(image_ids) == ground_truth.shape[0]
    print("Sampled %d tiles belonging to the video. Retaining all ground truth positives." % len(image_ids))
    print("Among them %d are ground truth positives. Does it look right?" % np.count_nonzero(ground_truth))

    softmax, X_all = _load_infer_result_and_pre_logit_redis(image_ids, redis_host)
    predictions = np.argmax(softmax, axis=1)
    assert X_all.shape[0] == predictions.shape[0]
    assert predictions.shape == ground_truth.shape
    print "Confusion matrix"
    print confusion_matrix(y_true=ground_truth, y_pred=predictions)

    # IMPORTANT: retain the imageid order!
    tp_mask = np.logical_and(predictions == 1, ground_truth == 1)
    fp_mask = np.logical_and(predictions == 1, ground_truth == 0)
    print("Found %d TPs, %d FPs. Does it look right?" % (np.count_nonzero(tp_mask), np.count_nonzero(fp_mask)))

    y_all_tp = np.zeros_like(predictions)
    y_all_tp[tp_mask] = 1

    # sample FPs
    max_samples = np.count_nonzero(tp_mask)
    p = min(1.0, float(max_samples) / np.count_nonzero(fp_mask))
    fp_mask = np.logical_and(fp_mask, np.random.choice([True, False],
                                                       size=fp_mask.shape,
                                                       p=[p, 1.0 - p]))
    print("Sampled %d FPs" % np.count_nonzero(fp_mask))

    out_flags = np.logical_or(tp_mask, fp_mask)
    X_out = X_all[out_flags]
    y_out = y_all_tp[out_flags]

    print X_out.shape, y_out.shape

    if output_file is not None:
        dct = {'X': X_out, 'y': y_out, 'image_ids': itemgetter(*out_flags.tolist())(image_ids)}
        pickle.dump(dct,
                    open(output_file, 'wb'))


def filename2imageid(file_name):
    return os.path.splitext(os.path.basename(file_name))[0]


def _load_infer_result_and_pre_logit_redis(image_ids, redis_host):
    r_server = redis.StrictRedis(host=redis_host, port=6379, db=2)
    n = len(image_ids)
    softmax = np.empty((n, 2), dtype=np.float)
    pre_logit = np.empty((n, 1024), dtype=np.float)

    print("Loading %d inference results from Redis ..." % n)
    for j, imgid in enumerate(image_ids):
        res = r_server.get(imgid)
        assert isinstance(res, str)
        res = ast.literal_eval(res)
        assert isinstance(res, list)
        softmax[j, :] = np.array(res[:2])
        pre_logit[j, :] = np.array(res[2:])

        if j % 1000 == 0:
            print("\rLoaded %d" % j),

    print ""
    print("Done.")
    return softmax, pre_logit


def load_pre_logit_Xy(pre_logit_files):
    pickle_files = glob.glob(pre_logit_files)
    print("Loaded %d files: \n\t%s" % (len(pickle_files), '\n\t'.join(pickle_files)))
    inputs = [pickle.load(open(f, 'rb')) for f in pickle_files]
    X = np.concatenate([d['X'] for d in inputs])
    y = np.concatenate([d['y'] for d in inputs])
    image_ids = itertools.chain.from_iterable([d['image_ids'] for d in inputs])
    return X, y, image_ids


if __name__ == '__main__':
    fire.Fire()
