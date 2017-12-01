import fire
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight
from prepare_jit_train_data import load_pre_logit_Xy


def visualize(pre_logit_files):
    # visualize the temporal locality of event frames
    X, y, _ = load_pre_logit_Xy(pre_logit_files)
    print ''.join('*' if t == 1 else '-' for t in y)


# CLASSIFIER_CLS = SVC
CLASSIFIER_CLS = SGDClassifier


def train(pre_logit_files,
          save_model_path=None,
          test_ratio=0.1,
          split_pos=True,
          downsample_train=1.0,
          verbose=False):
    """
    :param downsample_train: down sample the training split.
    :param split_pos: if true, we split the data according to the ratio of positive/negative separately,
        instead of ratio of all samples.
    :param pre_logit_files:
    :param save_model_path:
    :param eval_every_iters:
    :param n_iters:
    :param test_ratio:
    :param shuffle: If false, sample order is retained when splitting.
    :param verbose:
    :return:
    """
    X, y, _ = load_pre_logit_Xy(pre_logit_files)
    assert X.shape[0] == y.shape[0]
    n_all = y.shape[0]

    if not split_pos:
        n_test = int(n_all * test_ratio)
        X_train, X_test = X[: -n_test], X[-n_test:]
        y_train, y_test = y[: -n_test], y[-n_test:]
    else:
        print "Splitting train/validation for positive/negative respectively."
        X_pos, y_pos = X[y == 1], y[y == 1]
        X_neg, y_neg = X[y == 0], y[y == 0]

        n_test_pos = int(X_pos.shape[0] * test_ratio)
        n_test_neg = int(X_neg.shape[0] * test_ratio)

        X_pos_train, X_pos_test = X_pos[: -n_test_pos], X_pos[-n_test_pos:]
        y_pos_train, y_pos_test = y_pos[: -n_test_pos], y_pos[-n_test_pos:]

        X_neg_train, X_neg_test = X_neg[: -n_test_neg], X_neg[-n_test_neg:]
        y_neg_train, y_neg_test = y_neg[: -n_test_neg], y_neg[-n_test_neg:]

        X_train = np.concatenate([X_pos_train, X_neg_train], axis=0)
        y_train = np.concatenate([y_pos_train, y_neg_train])

        X_test = np.concatenate([X_pos_test, X_neg_test], axis=0)
        y_test = np.concatenate([y_pos_test, y_neg_test])

    # Downsample training set
    n_train = int(X_train.shape[0] * downsample_train)
    if not split_pos:
        X_train, y_train = resample(X_train, y_train, n_samples=n_train, random_state=42)
    else:
        X_pos_train, y_pos_train = resample(X_pos_train, y_pos_train, n_samples=max(n_train / 2, 1), random_state=42)
        X_neg_train, y_neg_train = resample(X_neg_train, y_neg_train, n_samples=max(n_train / 2, 1), random_state=42)
        X_train = np.concatenate([X_pos_train, X_neg_train], axis=0)
        y_train = np.concatenate([y_pos_train, y_neg_train])

    assert X_train.shape[1] == X_test.shape[1] == 1024
    print "All: %d / %d" % (y.shape[0], np.count_nonzero(y))
    print "Train set: %d / %d" % (y_train.shape[0], np.count_nonzero(y_train))
    print "Test set: %d / %d" % (y_test.shape[0], np.count_nonzero(y_test))

    clf = CLASSIFIER_CLS(random_state=42,
                         verbose=verbose,
                         class_weight='balanced')

    clf.fit(X_train, y_train)

    print "Final train accuracy: %f" % clf.score(X_train, y_train)
    print "Final test accuracy: %f " % clf.score(X_test, y_test)

    print "Confusion matrix on test:"
    pred_test = clf.predict(X_test)
    cm = confusion_matrix(y_true=y_test, y_pred=pred_test)
    print cm

    if save_model_path is not None:
        print "saving model to " + save_model_path
        pickle.dump(clf, open(save_model_path, 'wb'))


def plot_frame_accuracy(input_file, savefig=None):
    df = pd.read_csv(
        input_file,
        sep=r'\s+'
    )
    print df

    xlabels = map(int, df.columns[2:])
    for _, row in df.iterrows():
        x = xlabels
        y = np.array(row[2:])
        print x,y
        plt.plot(xlabels, np.array(row[2:]), '-o')

    plt.axis([0, max(xlabels), 0, 1.0])
    # plt.show()

    if savefig:
        plt.savefig(savefig)


if __name__ == '__main__':
    fire.Fire()
