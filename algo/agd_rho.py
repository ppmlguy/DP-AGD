import argparse
import numpy as np
import time
import math
from agd.common.dat import load_dat
# from agd.common.logistic import logistic_grad
# from agd.common.logistic import logistic_loss
# from agd.common.logistic import logistic_test
from agd.common.svm import svm_grad
from agd.common.svm import svm_loss
from agd.common.svm import svm_test
from agd.common.noisy_max import noisy_max
from agd.common.param import compute_sigma
from agd.common.param import dp_to_zcdp


def grad_avg(rho_old, rho_H, true_grad, noisy_grad, grad_clip):
    dim = true_grad.shape[0]
    sigma = grad_clip / math.sqrt(2.0 * (rho_H - rho_old))

    # new estimate
    g_2 = true_grad + sigma * np.random.randn(dim)

    beta = rho_old / rho_H
    # weighted average
    s_tilde = beta * noisy_grad + (1.0 - beta) * g_2

    return s_tilde


def agd_rho(X, y, rho, eps_total, delta, grad_func, loss_func, test_func,
            obj_clip, grad_clip, reg_coeff=0.0, batch_size=-1, exp_dec=1.0,
            gamma=0.1, splits=60, verbose=False):
    N, dim = X.shape

    # parameters
    eps_nmax = (eps_total * 0.5) / splits
    sigma = compute_sigma(eps_nmax, delta, grad_clip)

    # intial privacy budget
    rho_nmax = 0.5 * (eps_nmax**2)
    rho_ng = (grad_clip**2) / (2.0 * sigma**2)
    # rho_ng = rho_nmax

    # initialize the parameter vector
    w = np.zeros(dim)
    t = 0
    chosen_step_sizes = []
    max_step_size = 2.0
    n_candidate = 20

    while rho > 0:
        if verbose:
            loss = loss_func(w, X, y) / N
            acc = test_func(w, X, y)
            print "[{}] loss: {:.5f}  acc: {:5.2f}".format(t, loss, acc*100)

        if batch_size > 0:
            # build a mini-batch
            rand_idx = np.random.choice(N, size=batch_size, replace=False)
            mini_X = X[rand_idx, :]
            mini_y = y[rand_idx]
        else:
            mini_X = X
            mini_y = y

        # non-private (clipped) gradient
        grad = grad_func(w, mini_X, mini_y, grad_clip)

        sigma = grad_clip / math.sqrt(2.0 * rho_ng)
        noisy_grad = grad + sigma * np.random.randn(dim)
        noisy_unnorm = np.copy(noisy_grad)
        noisy_grad /= np.linalg.norm(noisy_grad)

        rho -= rho_ng

        # regularization
        if reg_coeff > 0:
            noisy_grad += reg_coeff * w

        idx = 0

        while idx == 0:
            # test if this is a descent direction
            step_sizes = np.linspace(0, max_step_size, n_candidate+1)
            candidate = [w - step * noisy_grad for step in step_sizes]
            scores = [loss_func(theta, mini_X, mini_y, clip=obj_clip)
                      for theta in candidate]
            scores[0] *= exp_dec

            # deduct the privacy budget used for noisy max
            lmbda = obj_clip/math.sqrt(2.0 * rho_nmax)
            idx, _ = noisy_max(scores, lmbda, bmin=True)
            rho -= rho_nmax

            # used up the budget
            if rho < 0:
                break

            if idx > 0:
                # don't do the update when the remain budget is insufficient
                if rho >= 0:
                    w[:] = candidate[idx-1]
                rho -= rho_ng
            else:
                rho_old = rho_ng
                rho_ng *= (1.0 + gamma)
                # max_step_size *= 0.9
                if verbose:
                    print "\tbad gradient: resample (rho_ng={})".format(rho_ng)

                noisy_grad = grad_avg(rho_old, rho_ng, grad, noisy_unnorm,
                                      grad_clip)
                noisy_grad /= np.linalg.norm(noisy_grad)
                rho -= (rho_ng - rho_old)

                if reg_coeff > 0:
                    noisy_grad += reg_coeff * w

        chosen_step_sizes.append(step_sizes[idx])
        t += 1

        if (t % 10) == 0:
            max_step_size = min(1.1*max(chosen_step_sizes), 2.0)
            del chosen_step_sizes[:]

    return w


def main(args):
    # load the dataset
    fpath = "../../../Experiment/Dataset/dat/{0}.dat".format(args.dname)
    X, y = load_dat(fpath, minmax=(0, 1), normalize=False, bias_term=True)
    N = X.shape[0]
    eps_total = args.eps
    delta = args.delta
    obj_clip = args.obj_clip
    grad_clip = args.grad_clip

    # for svm only
    y[y < 1] = -1

    rho = dp_to_zcdp(eps_total, delta)
    print "rho = {:.5f}".format(rho)

    start_time = time.clock()
    w = agd_rho(X, y, rho, eps_total, delta, svm_grad, svm_loss,
                svm_test, obj_clip, grad_clip, reg_coeff=0.01,
                exp_dec=args.exp_dec,
                verbose=True)
    print "time = ", time.clock() - start_time
    loss = svm_loss(w, X, y) / N
    acc = svm_test(w, X, y)
    print "loss: {:.5f}\t  acc: {:5.2f}".format(loss, acc*100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='adaptive sgd')
    parser.add_argument('dname', help='dataset name')
    parser.add_argument('eps', type=float, help='privacy budget')
    parser.add_argument('--delta', type=float, default=1e-8)
    parser.add_argument('--grad_clip', type=float, default=3.0)
    parser.add_argument('--obj_clip', type=float, default=3.0)
    parser.add_argument('--exp_dec', type=float, default=1.0)

    args = parser.parse_args()
    print "Parameters"
    print "----------"

    for arg in vars(args):
        print " - {0:22s}: {1}".format(arg, getattr(args, arg))

    main(args)
