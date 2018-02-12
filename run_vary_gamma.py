"""
Varying the privacy budget for NoisyMax
"""
import argparse
import numpy as np
from agd.common.dat import load_dat
from agd.common.logistic import logistic_grad
from agd.common.logistic import logistic_loss
from agd.common.logistic import logistic_test
from agd.common.param import dp_to_zcdp
from agd.algo.agd_rho import agd_rho as agd


def main(args):
    # load the dataset
    fpath = "./dataset/{0}.dat".format(args.dname)
    X, y = load_dat(fpath, minmax=(0, 1), normalize=False, bias_term=True)
    N = X.shape[0]

    delta = args.delta
    obj_clip = args.obj_clip
    grad_clip = args.grad_clip

    # variables to change
    epsilon = [0.1, 0.2, 0.4, 0.8, 1.6]
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5]
    n_eps = len(epsilon)
    n_rep = 10
    n_gammas = len(gammas)

    loss = np.zeros((n_gammas, n_eps, n_rep))
    acc = np.zeros_like(loss)

    for i, gamma in enumerate(gammas):
        for j, eps in enumerate(epsilon):
            rho = dp_to_zcdp(eps, delta)
            print "rho = {:.5f}".format(rho)

            for k in range(n_rep):
                w = agd(X, y, rho, eps, delta, logistic_grad, logistic_loss,
                        logistic_test, obj_clip, grad_clip, reg_coeff=0.0,
                        gamma=gamma)
                loss[i, j, k] = logistic_loss(w, X, y) / N
                acc[i, j, k] = logistic_test(w, X, y)

    avg_loss = np.mean(loss, axis=2)
    avg_acc = np.mean(acc, axis=2)

    np.savetxt('varying_gamma_{0}_acc.out'.format(args.dname),
               avg_acc, fmt='%.5f')
    np.savetxt('varying_gamma_{0}_obj.out'.format(args.dname),
               avg_loss, fmt='%.5f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='adaptive sgd')
    parser.add_argument('dname', help='dataset name')
    parser.add_argument('--delta', type=float, default=1e-8)
    parser.add_argument('--grad_clip', type=float, default=3.0)
    parser.add_argument('--obj_clip', type=float, default=3.0)

    args = parser.parse_args()
    print "Parameters"
    print "----------"

    for arg in vars(args):
        print " - {0:22s}: {1}".format(arg, getattr(args, arg))

    main(args)
