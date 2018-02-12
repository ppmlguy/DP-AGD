import numpy as np


def svm_loss(w, X, y, clip=-1):
    is_1d = w.ndim == 1

    w = np.atleast_2d(w)
    y = np.atleast_2d(y)

    wx = np.dot(X, w.T)
    obj = 1.0 - (y.T * wx)
    obj[obj < 0] = 0

    # clipping
    if clip > 0:
        obj[obj > clip] = clip

    # reg = lmbda * np.sum(np.square(w[:, 1:]), axis=1)
    loss = np.sum(obj, axis=0)

    # loss = hinge + reg

    if is_1d:
        loss = np.asscalar(loss)

    return loss


def svm_grad(w, X, y, clip=-1):
    y2d = np.atleast_2d(y)

    ywx = y * np.dot(X, w)
    loc = ywx < 1
    per_grad = -1.0 * y2d[:, loc].T * X[loc]

    if clip > 0:
        norm = np.linalg.norm(per_grad, axis=1)
        to_clip = norm > clip
        per_grad[to_clip, :] = ((clip * per_grad[to_clip])
                                / np.atleast_2d(norm[to_clip]).T)
        grad = np.sum(per_grad, axis=0)
    else:
        grad = np.sum(per_grad, axis=0)

    return grad


def svm_loss_and_grad(w, X, y, reg_coeff=0.0001):
    N = X.shape[0]

    y2d = np.atleast_2d(y)

    # loss
    wx = np.dot(X, w)
    reg = 0.5 * reg_coeff * np.dot(w, w)

    margin = 1. - (y * wx)
    loc = margin < 0
    margin[loc] = 0

    loss = reg + (np.sum(margin, axis=0) / float(N))

    # gradient
    grad = reg_coeff * w
    yx = -1. * (y2d.T * X)
    dh = np.sum(yx[~loc], axis=0) / float(N)
    grad += dh

    return loss, grad


def svm_test(w, X, y):
    is_1d = w.ndim == 1

    N = X.shape[0]
    w2d = np.atleast_2d(w)
    y2d = np.atleast_2d(y)

    wx = np.dot(X, w2d.T)
    sign = y2d.T * wx

    cnt = np.count_nonzero(sign > 0, axis=0)

    if is_1d:
        cnt = np.squeeze(cnt)

    return cnt / float(N)
