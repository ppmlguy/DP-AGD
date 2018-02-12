import argparse
import numpy as np
from agd.common.svm import svm_grad
from agd.common.svm import svm_loss
from agd.common.svm import svm_test
from agd.common.gaussian_moments import compute_log_moment
from agd.common.gaussian_moments import get_privacy_spent
from agd.common.param import compute_advcomp_budget
from agd.common.param import compute_sigma
from agd.common.dat import load_dat


def dpsgd_ma(X, y, grad, sigma, T, step_size, batch_size, clip=4, delta=1e-8,
             reg_coeff=0.0):
    N, dim = X.shape
    n = N * 1.0

    # initialize the parameter vector
    w = np.zeros(dim)
    q = batch_size / n

    # moments accountant
    max_lmbd = 32
    log_moments = []
    for lmbd in xrange(1, max_lmbd+1):
        log_moment = compute_log_moment(q, sigma, T, lmbd)
        log_moments.append((lmbd, log_moment))

    eps, _ = get_privacy_spent(log_moments, target_delta=delta)

    for t in range(T):
        # build a mini-batch
        rand_idx = np.random.choice(N, size=batch_size, replace=False)
        mini_X = X[rand_idx, :]
        mini_y = y[rand_idx]

        gt = grad(w, mini_X, mini_y, clip=clip)
        gt += (sigma * clip) * np.random.randn(dim)
        gt /= batch_size

        # regularization
        gt += reg_coeff * w

        w -= step_size * gt

    return w, eps


def dpsgd_adv(X, y, grad, eps, T, step_size, batch_size, clip=3, delta=1e-8,
              reg_coeff=0.001):
    N, dim = X.shape
    n = N * 1.0

    eps_iter, delta_iter = compute_advcomp_budget(eps, delta, T)

    # initialization
    w = np.zeros(dim)
    q = batch_size / n

    # privacy amplification by sampling
    # (e, d)-DP => (2qe, d)-DP
    eps_iter /= 2.0 * q
    sigma = compute_sigma(eps_iter, delta_iter, 2.0*clip)

    for t in range(T):
        # build a mini-batch
        rand_idx = np.random.choice(N, size=batch_size, replace=False)
        mini_X = X[rand_idx, :]
        mini_y = y[rand_idx]

        gt = grad(w, mini_X, mini_y, clip=clip)
        gt += sigma * np.random.randn(dim)
        gt /= batch_size

        gt += reg_coeff * w

        w -= step_size * gt

    return w


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='adaptive sgd')
    parser.add_argument('dname', help='dataset name')

    args = parser.parse_args()

    # load the dataset
    fpath = "../../../Experiment/Dataset/dat/{0}.dat".format(args.dname)
    X, y = load_dat(fpath, minmax=(0, 1), normalize=False, bias_term=True)
    y[y < 1] = -1

    N, dim = X.shape

    sigma = 4
    batch_size = 1000
    learning_rate = 0.05
    reg_coeff = 0.001

    print "SGD with moments accountant"
    for T in [1, 100, 1000, 10000, 20000]:
        w, eps = dpsgd_ma(X, y, svm_grad, sigma, T, learning_rate,
                          batch_size, reg_coeff=reg_coeff)
        loss = svm_loss(w, X, y) / N
        acc = svm_test(w, X, y)

        print "[T={:5d}] eps: {:.5f}\tloss: {:.5f}\tacc: {:5.2f}".format(
            T, eps, loss, acc*100)

    print "\nSGD with advanced composition"
    for eps in [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]:
        # used the same heuristic as in PrivGene
        T = max(int(round((N * eps) / 500.0)), 1)
        w = dpsgd_adv(X, y, svm_grad, eps, T, 0.1, batch_size)
        loss = svm_loss(w, X, y) / N
        acc = svm_test(w, X, y)

        print "eps: {:4.2f}\tloss: {:.5f}\tacc: {:5.2f}".format(
            eps, loss, acc*100)
