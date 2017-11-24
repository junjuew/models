import fire
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from prepare_jit_train_data import load_pre_logit_Xy


def visualize(pre_logit_files):
    # visualize the temporal locality of event frames
    X, y, _ = load_pre_logit_Xy(pre_logit_files)
    print ''.join('*' if t == 1 else '-' for t in y)


def train(pre_logit_files,
          save_model_path=None,
          eval_every_iters=10,
          n_iters=50,
          test_ratio=0.1,
          shuffle=False,
          verbose=False):
    """
    Example:
    python svm_on_pre_logit.py train
        --pre_logit_files /home/zf/opt/drone-scalable-search/processed_dataset/stanford_campus/experiments/tiled_mobilenet_classification/2_more_test/bookstore_video0_tp_fp.p
        --test_ratio 0.3
        --eval_every_iters=10
    :param pre_logit_files:
    :param save_model_path:
    :param eval_every_iters:
    :param n_iters:
    :param test_ratio:
    :param shuffle:
    :param verbose:
    :return:
    """
    X, y, _ = load_pre_logit_Xy(pre_logit_files)

    positive_ratio = (float(np.count_nonzero(y)) / y.shape[0])
    print ""
    print "Positive ratio: %f" % positive_ratio
    if abs(positive_ratio - 0.5) >= 0.2:
        print "This is quite imbalanced. Are you sure?"
        print "(Consider increasing the over_sample_ratio when constructing the dataset)"
    print ""

    if shuffle:
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=test_ratio, random_state=42)
    else:
        split_point = int(y.shape[0] * test_ratio)
        X_train, X_validation = X[: -split_point], X[-split_point:]
        y_train, y_validation = y[: -split_point], y[-split_point:]

    for var in X_train, X_validation, y_train, y_validation:
        print var.shape

    print "Train set: %d / %d" % (y_train.shape[0], np.count_nonzero(y_train))
    print "Validation set: %d / %d" % (y_validation.shape[0], np.count_nonzero(y_validation))

    accuracies = []
    clf = SGDClassifier(random_state=42, verbose=verbose)  # TODO grid search hyperparameters

    for i in range(n_iters):
        clf.partial_fit(X_train, y_train, classes=np.array([0, 1]))

        if i % eval_every_iters == 0:
            train_acc = clf.score(X_train, y_train)
            valid_acc = clf.score(X_validation, y_validation)
            accuracies.append((i, train_acc, valid_acc))
            print "%f\t%f" % (train_acc, valid_acc)

    print "Finished training"

    print [t[1] for t in accuracies]
    print [t[2] for t in accuracies]

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


def retrain(pre_logit_files,
            checkpoint_path,
            save_model_path=None,
            eval_every_iters=20,
            n_iter=10):
    X_new, y_new, _ = load_pre_logit_Xy(pre_logit_files)
    X_train, X_validation, y_train, y_validation = \
        train_test_split(X_new, y_new, test_size=0.1, random_state=42)

    print X_train[:3], y_train[:13]

    clf = pickle.load(open(checkpoint_path, 'rb'))

    before_train_acc = clf.score(X_train, y_train)
    before_valid_acc = clf.score(X_validation, y_validation)

    accuracies = []
    for i in range(n_iter):
        clf.partial_fit(X_train, y_train)
        if i % eval_every_iters == 0:
            train_acc = clf.score(X_train, y_train)
            valid_acc = clf.score(X_validation, y_validation)
            accuracies.append((i, train_acc, valid_acc))
            print "Train acc = %f,\nvalidation acc = %f" % (train_acc, valid_acc)

    print [t[1] for t in accuracies]
    print [t[2] for t in accuracies]

    print "train accuracy before retrain: ", before_train_acc
    print "validation accuracy before retrain: ", before_valid_acc
    print "Final train accuracy after retrain: ", clf.score(X_train, y_train)
    print "Final validation accuracy after retrain: ", clf.score(X_validation, y_validation)

    if save_model_path is not None:
        print "saving model to " + save_model_path
        pickle.dump(clf, open(save_model_path, 'wb'))


if __name__ == '__main__':
    fire.Fire()
