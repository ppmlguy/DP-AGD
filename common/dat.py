import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer


def majority_label(y):
    N = y.shape[0]

    labels = np.unique(y)
    n_class = labels.shape[0]

    label_cnt = np.zeros(n_class)
    cnt_sum = 0

    for i in range(n_class-1):
        label_cnt[i] = np.count_nonzero(y == labels[i])
        cnt_sum += label_cnt[i]

    label_cnt[n_class-1] = N - cnt_sum
    label_cnt /= float(N)

    return max(label_cnt)


def load_dat(filepath, minmax=None, normalize=False, bias_term=True):
    """ load a dat file

    args:
    minmax: tuple(min, max), dersired range of transformed data
    normalize: boolean, normalize samples individually to unit norm if True
    bias_term: boolean, add a dummy column of 1s
    """
    lines = np.loadtxt(filepath)
    labels = lines[:, -1]
    features = lines[:, :-1]

    N, dim = features.shape

    if minmax is not None:
        minmax = MinMaxScaler(feature_range=minmax, copy=False)
        minmax.fit_transform(features)

    if normalize:
        # make sure each entry's L2 norm is 1
        normalizer = Normalizer(copy=False)
        normalizer.fit_transform(features)

    if bias_term:
        X = np.hstack([np.ones(shape=(N, 1)), features])
    else:
        X = features

    return X, labels


def main():
    X, y = load_dat("../../Dataset/dat/IPUMS-BR.dat", bias_term=False)

    norm = np.linalg.norm(X, axis=1)
    print "# of examples whose norm is already < 1 : ", \
        np.count_nonzero(norm < 1)

    print "Min values"
    print X.min(axis=0)

    print "\nMax values"
    print X.max(axis=0)
    print "l2 norm=", np.linalg.norm(X[:5, 1:], axis=1)


if __name__ == "__main__":
    main()
