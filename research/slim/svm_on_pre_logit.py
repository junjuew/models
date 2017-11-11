import fire
import glob
import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def train(dataset_dir, prefix='pre_logit_', eval_every_steps=20, max_steps=-1):
    path = os.path.join(dataset_dir, prefix + '*.p')
    pickle_files = glob.glob(path)
    inputs = [pickle.load(open(f, 'rb')) for f in pickle_files]

    print("Loaded %d files" % len(inputs))

    X = np.concatenate([d['X'] for d in inputs])
    y = np.concatenate([d['y'] for d in inputs])

    print("Loaded X " + str(X.shape))
    print("Loaded y " + str(y.shape))

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.1, random_state=42)

    for var in X_train, X_validation, y_train, y_validation:
        print var.shape

    print X_train[0], y_train[0]

    clf = SVC(verbose=True)
    clf.fit(X_train, y_train)

    print "Finished training"
    print "train accuracy: %f" % clf.score(X_train, y_train)
    print "validation accuracy: %f " % clf.score(X_validation, y_validation)


if __name__ == '__main__':
    fire.Fire()
