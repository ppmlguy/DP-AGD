import numpy as np
from jwu.util.extmath import log_logistic


def logistic_score(X, y, w, C=1):
    y2d = np.atleast_2d(y)

    t = np.dot(X, w.T)
    score = np.sum(y2d.T * t, axis=0)
    # score -= np.sum(np.log(1 + np.exp(t)), axis=0)
    score += np.sum(log_logistic(-t), axis=0)

    return score


def svm_score(X, y, w, C=10):
    y2d = np.atleast_2d(y)

    wx = np.dot(X, w.T)
    hinge = 1 - (y2d.T * wx)
    hinge[hinge < 0] = 0

    reg = np.sum(np.square(w[:, 1:]), axis=1)
    score = C*np.sum(hinge, axis=0) + 0.5*reg

    return -score
