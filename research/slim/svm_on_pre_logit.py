import fire
import glob
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

DEFAULT_PREFIX = 'pre_logit_'


def train(dataset_dir,
          prefix=DEFAULT_PREFIX,
          save_model_path=None,
          n_iters=20):
    X, y = _load_pre_logit_Xy(dataset_dir, prefix)

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1, random_state=42)

    for var in X_train, X_validation, y_train, y_validation:
        print var.shape

    print X_train[:3], y_train[:3]

    # scaler = StandardScaler(with_mean=True, with_std=True)

    clf = SGDClassifier(random_state=42, verbose=True)  # TODO grid search hyperparameters
    for _ in range(n_iters):
        clf.partial_fit(X_train, y_train, classes=np.array([0, 1]))

    print "Finished training"
    print "train accuracy: %f" % clf.score(X_train, y_train)
    print "validation accuracy: %f " % clf.score(X_validation, y_validation)

    if save_model_path is not None:
        print "saving model to " + save_model_path
        pickle.dump(clf, open(save_model_path, 'wb'))


def eval(dataset_dir,
         checkpoint_path,
         prefix=DEFAULT_PREFIX):
    X_eval, y_eval = _load_pre_logit_Xy(dataset_dir, prefix)

    clf = pickle.load(open(checkpoint_path, 'rb'))

    score = clf.score(X_eval, y_eval)
    return score


def retrain(dataset_dir,
            checkpoint_path,
            save_model_path=None,
            prefix=DEFAULT_PREFIX,
            n_iter=10):
    X_new, y_new = _load_pre_logit_Xy(dataset_dir, prefix)
    X_train, X_validation, y_train, y_validation = \
        train_test_split(X_new, y_new, test_size=0.1, random_state=42)

    print X_train[:3], y_train[:13]

    clf = pickle.load(open(checkpoint_path, 'rb'))

    print "train accuracy before retrain: ", clf.score(X_train, y_train)
    print "validation accuracy before retrain: ", clf.score(X_validation, y_validation)
    for _ in range(n_iter):
        clf.partial_fit(X_train, y_train)
    print "train accuracy after retrain: ", clf.score(X_train, y_train)
    print "validation accuracy after retrain: ", clf.score(X_validation, y_validation)

    if save_model_path is not None:
        print "saving model to " + save_model_path
        pickle.dump(clf, open(save_model_path, 'wb'))


def _load_pre_logit_Xy(dataset_dir, prefix):
    wildcard = os.path.join(dataset_dir, prefix + '*.p')
    pickle_files = glob.glob(wildcard)
    print("Loaded %d files: \n\t%s" % (len(pickle_files), '\n\t'.join(pickle_files)))
    inputs = [pickle.load(open(f, 'rb')) for f in pickle_files]
    X = np.concatenate([d['X'] for d in inputs])
    y = np.concatenate([d['y'] for d in inputs])
    print("Loaded X " + str(X.shape))
    print("Loaded y " + str(y.shape))
    return X, y


if __name__ == '__main__':
    fire.Fire()
