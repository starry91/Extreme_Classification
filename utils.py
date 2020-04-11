import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import pickle
import pandas as pd
from scipy.io.arff import loadarff
from xclib.data import data_utils
import math


def p_k(y_predicted, y_actual, k):
    res = 0
    for i in range(len(y_predicted)):
        indices = y_predicted[i].argsort()[-k:][::-1]
        res += (y_actual[i][indices].sum())/k
    return res/len(y_predicted)


def dcg_k(y_predicted, y_actual, k):
    def logger(t): return 1/math.log(1+t, 2)
    logger = np.vectorize(logger)
    # l = np.arange(1,1+y_predicted.shape[1])
    l = np.arange(1, 1+k)
    l = logger(l)
    res = np.zeros(y_predicted.shape[0])
    for i in range(len(y_predicted)):
        indices = y_predicted[i].argsort()[-k:][::-1]
        temp = l*y_actual[i][indices]
        res[i] = temp.sum()
        # temp = l*y_actual[i]
        # res[i] = temp[indices].sum()
    return res


def n_k(y_predicted, y_actual, k):
    dcg_k_list = dcg_k(y_predicted, y_actual, k)
    def logger(t): return 1/math.log(1+t, 2)
    logger = np.vectorize(logger)
    res = 0
    for i in range(len(y_predicted)):
        lim = y_actual[i].sum()
        if(lim == 0):
            continue
        l = np.arange(1, 1+min(lim, k))
        l = logger(l)
        deno = l.sum()
        res += dcg_k_list[i]/deno
    return res/y_predicted.shape[0]


def get_matrix_from_txt(path):
    features, labels, num_samples, num_features, num_labels = data_utils.read_data(
        path)
    return features.toarray(), labels.toarray().astype(int)


def get_data(path):
    raw_data = loadarff(path)
    df_data = pd.DataFrame(raw_data[0])
    X_cols = df_data.columns[:120]
    Y_cols = df_data.columns[120:]
    X_data = df_data[X_cols]
    Y_data = df_data[Y_cols]
    for col in Y_data.columns:
        Y_data[col] = Y_data[col].apply(lambda x: x.decode("utf-8"))
        Y_data[col] = Y_data[col].astype(int)
    return X_data.values, Y_data.values


def load_data(path, isTxt=False):
    """loads the data and converts to numpy arrays"""
    print('loading data ...')

    if(not isTxt):
        X_train, Y_train = get_data(path)
    else:
        X_train, Y_train = get_matrix_from_txt(path)
    return X_train, Y_train


def make_tensor(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = torch.tensor(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret
