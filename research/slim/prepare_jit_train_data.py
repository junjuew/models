import pickle

import fire
import glob

import itertools
import numpy as np
from svm_on_pre_logit import load_pre_logit_Xy

def make_tp_fp_pre_logit_data(pre_logit_files, inference_files)
    pre_logit_files = glob.glob(pre_logit_files)
    X, y = load_pre_logit_Xy(pre_logit_files)
    inference_files = glob.glob(inference_files)





def _load_inference_result(result_files):
    result_files = glob.glob(result_files)
    print("Loading %d result files: \n\t%s" % (len(result_files), '\n\t'.join(result_files)))
    inputs = [pickle.load(open(f, 'rb')) for f in result_files]
    image_ids = itertools.chain.from_iterable([d['image_dis'] for d in inputs])
    predictions = np.concatenate([d['predictions'] for d in inputs])
    assert len(image_ids) == predictions.shape[0]
    print("Loaded %d inference results." % len(predictions))
    return image_ids, predictions


if __name__ == '__main__':
    fire.Fire()