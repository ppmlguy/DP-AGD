import argparse
import time
import numpy as np
from agd.common.dat import load_dat
from agd.common.logistic import logistic_grad
from agd.common.logistic import logistic_loss
from agd.common.logistic import logistic_test


def outpert_gd(X, y, grad, eps, T, L, step_size, delta=1e-8,
               reg_coeff=0.001, verbose=False):
    """
    Parameters:
    -------------
    X, y : input dataset
    grad : gradient function
    eps, delta: privacy parameters
    sens : sensitivity
    """
    N, dim = X.shape
    n = float(N)

    sens = (3.0 * L * T * step_size) / n

    if verbose:
        print "sensitivity={:.5f}".format(sens)

    # initialization
    w = np.zeros(dim)

    # run T-steps of full graident descent
    for t in range(T):
        gt = grad(w, X, y) / n
        gt += reg_coeff * w

        w -= step_size * gt

    # adding noise
    sigma = (np.sqrt(2.0 * np.log(2.0/delta)) * sens) / eps
    z = sigma * np.random.randn(dim)

    return w + z


def main(args):
    fpath = "../../../Experiment/Dataset/dat/{0}.dat".format(args.dname)
    X, y = load_dat(fpath, minmax=(0, 1), normalize=False, bias_term=True)
    N, dim = X.shape
    print "({}, {})".format(N, dim)

    delta = args.delta
    Ts = [20, 30, 40, 50, 60, 70]
    epsilon = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    L = args.L
    step_size = args.step_size
    nT = len(Ts)
    nrep = args.rep
    neps = len(epsilon)

    acc = np.zeros((nT, neps, nrep))
    loss = np.zeros_like(acc)

    for i, T in enumerate(Ts):
        for j, eps in enumerate(epsilon):
            for k in range(nrep):
                w_priv = outpert_gd(X, y, logistic_grad, eps, T, L,
                                    step_size,
                                    delta=delta, reg_coeff=args.reg_coeff)

                loss[i, j, k] = logistic_loss(w_priv, X, y) / N
                acc[i, j, k] = logistic_test(w_priv, X, y) * 100

    avg_obj = np.mean(loss, axis=2)
    avg_acc = np.mean(acc, axis=2)

    filename = "outpert_logres_{0}_T".format(args.dname)
    np.savetxt("{0}_acc.out".format(filename), avg_acc, fmt='%.5f')
    np.savetxt("{0}_obj.out".format(filename), avg_obj, fmt='%.5f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='adaptive sgd')
    parser.add_argument('dname', help='dataset name')
    parser.add_argument('L', type=float, help='lipschitz constant')
    parser.add_argument('--delta', type=float, default=1e-8)
    parser.add_argument('--reg_coeff', type=float, default=0.001)
    parser.add_argument('--step_size', type=float, default=0.1)
    parser.add_argument('--rep', type=int, default=10)

    args = parser.parse_args()

    print "Running the program ... [{0}]".format(
        time.strftime("%m/%d/%Y %H:%M:%S"))
    print "Parameters"
    print "----------"

    for arg in vars(args):
        print " - {0:22s}: {1}".format(arg, getattr(args, arg))

    main(args)
