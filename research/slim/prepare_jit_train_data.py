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


def make_tp_fp_dataset(file_glob, redis_host, output_file=None, max_samples=3000, over_sample_ratio=20):
    """
    Separate one video's INFERRED positive tiles into TP and FP.
    :param over_sample_ratio: Over sample in order to find sufficient number of FPs,
        while avoiding loading all inference results. In rare cases, you may need to increase this number and rerun.
    :param redis_host:
    :param output_file: a pickle file {'X': np.array, 'y': np.array, 'image_ids': tuple<str>}
    :param file_glob: the wildcard giving a video's all tiles. Ground truth is guessed from the file paths.
    :param max_samples: Maximum number of samples to retain for each category (TP/FP)
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

    print '\n'.join(map(str, image_ids_and_ground_truth[::len(image_ids_and_ground_truth)/10]))

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
    tp_flags = np.logical_and(predictions == 1, ground_truth == 1)
    fp_flags = np.logical_and(predictions == 1, ground_truth == 0)
    print("Found %d TPs, %d FPs. Does it look right?" % (np.count_nonzero(tp_flags), np.count_nonzero(fp_flags)))

    y_all_tp = np.zeros_like(predictions)
    y_all_tp[tp_flags] = 1

    # sample FPs
    max_samples = np.count_nonzero(tp_flags)
    p = min(1.0, float(max_samples) / np.count_nonzero(fp_flags))
    fp_flags = np.logical_and(fp_flags, np.random.choice([True, False],
                                                         size=fp_flags.shape,
                                                         p=[p, 1.0 - p]))
    print("Sampled %d FPs" % np.count_nonzero(fp_flags))

    out_flags = np.logical_or(tp_flags, fp_flags)
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
    print("Loaded X " + str(X.shape))
    print("Loaded y " + str(y.shape))
    return X, y, image_ids


if __name__ == '__main__':
    fire.Fire()
