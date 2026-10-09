import numpy as np
from sklearn.datasets import load_iris

N_TRAIN_PER_CLASS = 35
SEED = 0


def load_split():
    data = load_iris()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.int64)

    rng = np.random.default_rng(SEED)
    train_idx, test_idx = [], []
    for c in (0, 1, 2):
        idx = rng.permutation(np.flatnonzero(y == c))
        train_idx.append(idx[:N_TRAIN_PER_CLASS])
        test_idx.append(idx[N_TRAIN_PER_CLASS:])

    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def standardize(X_train, X_test):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    return (X_train - mean) / std, (X_test - mean) / std


def get_data():
    X_train, y_train, X_test, y_test = load_split()
    X_train, X_test = standardize(X_train, X_test)
    return X_train, y_train, X_test, y_test
