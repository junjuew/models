import fire
import glob
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

DEFAULT_PREFIX = 'pre_logit_'


def train(dataset_dir, prefix=DEFAULT_PREFIX, save_model_path=None):
    X, y = _load_pre_logit_Xy(dataset_dir, prefix)

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1, random_state=42)

    for var in X_train, X_validation, y_train, y_validation:
        print var.shape

    print X_train[:3], y_train[:3]

    clf = SVC(verbose=True) # TODO grid search hyperparameters
    clf.fit(X_train, y_train)

    print "Finished training"
    print "train accuracy: %f" % clf.score(X_train, y_train)
    print "validation accuracy: %f " % clf.score(X_validation, y_validation)

    if save_model_path is not None:
        print "saving model to " + save_model_path
        pickle.dump(clf, open(save_model_path, 'wb'))

    # return clf


def eval(dataset_dir, checkpoint_path, prefix=DEFAULT_PREFIX):
    X_eval, y_eval = _load_pre_logit_Xy(dataset_dir, prefix)

    clf = pickle.load(open(checkpoint_path, 'rb'))
    assert isinstance(clf, SVC)

    score = clf.score(X_eval, y_eval)
    return score


def retrain(dataset_dir, checkpoint_path, save_model_path=None, prefix=DEFAULT_PREFIX):
    X_new, y_new = _load_pre_logit_Xy(dataset_dir, prefix)
    X_train, X_validation, y_train, y_validation = \
        train_test_split(X_new, y_new, test_size=0.1, random_state=42)

    print X_train[:3], y_train[:13]

    clf = pickle.load(open(checkpoint_path, 'rb'))

    print "before retrain: ", clf.score(X_validation, y_validation)
    clf.fit(X_train, y_train)
    print "after retrain: ", clf.score(X_validation, y_validation)

    if save_model_path is not None:
        print "saving model to " + save_model_path
        pickle.dump(clf, open(save_model_path, 'wb'))

    # return clf


def _load_pre_logit_Xy(dataset_dir, prefix):
    wildcard = os.path.join(dataset_dir, prefix + '*.p')
    pickle_files = glob.glob(wildcard)
    print("Loaded %d files: %s" % (len(pickle_files), ','.join(pickle_files)))
    inputs = [pickle.load(open(f, 'rb')) for f in pickle_files]
    X = np.concatenate([d['X'] for d in inputs])
    y = np.concatenate([d['y'] for d in inputs])
    print("Loaded X " + str(X.shape))
    print("Loaded y " + str(y.shape))
    return X, y


if __name__ == '__main__':
    fire.Fire()
