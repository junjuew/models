import fire
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.utils.class_weight import compute_class_weight
from prepare_jit_train_data import load_pre_logit_Xy


def visualize(pre_logit_files):
    # visualize the temporal locality of event frames
    X, y, _ = load_pre_logit_Xy(pre_logit_files)
    print ''.join('*' if t == 1 else '-' for t in y)


def train(pre_logit_files,
          save_model_path=None,
          test_ratio=0.1,
          split_pos=False,
          downsample_train=1.0,
          shuffle=False,
          verbose=False):
    """
    :param downsample_train: down sample the training split.
    :param split_pos: if true, we split the data according to the ratio of positive samples, instead of all samples.
        Only use when shuffle is False.
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

    # positive_ratio = (float(np.count_nonzero(y)) / y.shape[0])
    # print ""
    # print "Positive ratio: %f" % positive_ratio
    # if abs(positive_ratio - 0.5) >= 0.2:
    #     print "This is quite imbalanced. Are you sure?"
    #     print "(Consider increasing the over_sample_ratio when constructing the dataset)"
    # print ""

    if shuffle:
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=test_ratio, random_state=42)
    elif not split_pos:
        split_point = int(y.shape[0] * test_ratio)
        X_train, X_validation = X[: -split_point], X[-split_point:]
        y_train, y_validation = y[: -split_point], y[-split_point:]
    else:
        print "Split train/validation according to positive ratio."
        pos_inds = np.where(y == 1)[0]
        test_split_ind = pos_inds[-int(len(pos_inds) * test_ratio)]
        train_split_ind = pos_inds[int(len(pos_inds) * (1.0 - test_ratio) * downsample_train)]
        assert train_split_ind <= test_split_ind
        X_train, y_train = X[:train_split_ind, :], y[:train_split_ind]
        X_validation, y_validation = X[test_split_ind:, :], y[test_split_ind:]

    print "All: %d / %d" % (y.shape[0], np.count_nonzero(y))
    print "Train set: %d / %d" % (y_train.shape[0], np.count_nonzero(y_train))
    print "Validation set: %d / %d" % (y_validation.shape[0], np.count_nonzero(y_validation))

    clf = SGDClassifier(random_state=42,
                        verbose=verbose,
                        class_weight='balanced')

    clf.fit(X_train, y_train)

    # for i in range(n_iters):
    #     clf.partial_fit(X_train, y_train, classes=np.array([0, 1]))
    #
    #     if i % eval_every_iters == 0:
    #         train_acc = clf.score(X_train, y_train)
    #         valid_acc = clf.score(X_validation, y_validation)
    #         accuracies.append((i, train_acc, valid_acc))
    #         print "%f\t%f" % (train_acc, valid_acc)

    print "Finished training"

    print "Final train accuracy: %f" % clf.score(X_train, y_train)
    print "Final validation accuracy: %f " % clf.score(X_validation, y_validation)

    print "Confusion matrix on validation:"
    pred_validation = clf.predict(X_validation)
    cm = confusion_matrix(y_true=y_validation, y_pred=pred_validation)
    print cm

    if save_model_path is not None:
        print "saving model to " + save_model_path
        pickle.dump(clf, open(save_model_path, 'wb'))


def eval(pre_logit_files,
         checkpoint_path):
    X_eval, y_eval, _ = load_pre_logit_Xy(pre_logit_files)

    clf = pickle.load(open(checkpoint_path, 'rb'))

    score = clf.score(X_eval, y_eval)
    return score


if __name__ == '__main__':
    fire.Fire()
