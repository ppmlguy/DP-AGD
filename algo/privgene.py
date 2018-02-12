import argparse
import time
from math import sqrt
import numpy as np
from pgene_score import logistic_score
from agd.common.noisy_max import exp_mech
from agd.common.dat import load_dat
from agd.common.logistic import logistic_test
from agd.common.param import dp_to_zcdp


def privgene(X, y, eps, r, score_func, C=1, batch_size=-1):
    """ PrivGene

    Parameters
    -----------
    r : number of iterations
    eps: per-iteration privacy budget, eps=sqrt(2*rho/r) => (rho/r)-zCDP
    C : sensitivity factor, fix C=1 for logsitic regression (see the paper)
    """
    N, dim = X.shape

    # mutation settings
    mut_scale = 0.5
    mut_resize = 0.95

    # initialize \Omega
    seed_remain = np.zeros((1, dim))
    cur_scale = mut_scale
    seed = np.zeros((2*dim, dim))

    for i in range(dim):
        seed[2*i, i] = cur_scale
        seed[2*i+1, i] = -1. * cur_scale

    for i in range(r):
        if batch_size > 0:
            # build a mini-batch
            rand_idx = np.random.choice(N, size=batch_size, replace=False)
            mini_X = X[rand_idx, :]
            mini_y = y[rand_idx]
        else:
            mini_X = X
            mini_y = y

        utility = score_func(mini_X, mini_y, seed, C)

        sens = 4. * C * cur_scale
        sigma = eps / sens

        selected, _ = exp_mech(utility, sigma)
        seed_remain = seed[selected, :]

        cur_scale *= mut_resize
        for j in range(dim):
            mutation = np.zeros(dim)
            mutation[j] = cur_scale
            seed[2*j, :] = seed_remain + mutation
            seed[2*j+1, :] = seed_remain - mutation

    return seed_remain


def main(args):
    # input parameters
    eps = args.eps
    delta = args.delta

    # load the data
    X, y = load_dat("../dataset/{0}.dat".format(args.dsname))

    N, dim = X.shape

    # number of iterations
    r = max(int(round((N * eps) / 800.0)), 1)

    # privacy budget
    rho = dp_to_zcdp(eps, delta)
    eps_iter = sqrt((2.*rho) / r)

    print " - {0:22s}: {1}".format("r", r)
    print " - {0:22s}: {1}".format("eps_iter", eps_iter)

    score_func = logistic_score
    C = 1
    sol = privgene(X, y, eps_iter, r, score_func, C=C)
    acc = logistic_test(sol, X, y)
    print "Accuracy=", acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='directional search')
    parser.add_argument('dsname', help='dataset name')
    parser.add_argument('eps', type=float, help='epsilon')
    parser.add_argument('delta', type=float, help='number of iterations')

    args = parser.parse_args()

    print "Running the program ... [{0}]".format(
        time.strftime("%m/%d/%Y %H:%M:%S"))
    print "Parameters"
    print "----------"
    for arg in vars(args):
        print " - {0:22s}: {1}".format(arg, getattr(args, arg))

    # main function
    main(args)
