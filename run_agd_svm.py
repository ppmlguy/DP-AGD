import argparse
import time
import numpy as np
from agd.common.dat import load_dat
from agd.common.svm import svm_grad
from agd.common.svm import svm_loss
from agd.common.svm import svm_test
from agd.common.param import dp_to_zcdp
from agd.algo.agd_rho import agd_rho as agd
from sklearn.model_selection import RepeatedKFold


def main(args):
    fpath = "./dataset/{0}.dat".format(args.dname)
    X, y = load_dat(fpath, minmax=(0, 1), normalize=False, bias_term=True)
    N, dim = X.shape

    y[y < 1] = -1

    nrep = args.rep
    delta = args.delta
    obj_clip = args.obj_clip
    grad_clip = args.grad_clip

    epsilon = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    neps = len(epsilon)

    K = 5  # 5-folds cross-validation
    cv_rep = 2
    k = 0
    acc = np.zeros((neps, nrep, K*cv_rep))
    obj = np.zeros((neps, nrep, K*cv_rep))

    rkf = RepeatedKFold(n_splits=K, n_repeats=cv_rep)

    for train, test in rkf.split(X):
        train_X, train_y = X[train, :], y[train]
        test_X, test_y = X[test, :], y[test]

        n_train = train_X.shape[0]

        for i, eps in enumerate(epsilon):
            rho = dp_to_zcdp(eps, delta)

            for j in range(nrep):
                sol = agd(train_X, train_y, rho, eps, delta, svm_grad,
                          svm_loss, svm_test, obj_clip, grad_clip,
                          reg_coeff=args.reg_coeff,
                          batch_size=args.batch_size,
                          exp_dec=args.exp_dec,
                          gamma=args.gamma)
                obj[i, j, k] = svm_loss(sol, train_X, train_y) / n_train
                acc[i, j, k] = svm_test(sol, test_X, test_y) * 100.0
                # print "acc[{},{},{}]={}".format(i, j, k, acc[i, j, k])

        k += 1

    avg_acc = np.vstack([np.mean(acc, axis=(1, 2)),
                         np.std(acc, axis=(1, 2))])
    avg_obj = np.vstack([np.mean(obj, axis=(1, 2)),
                         np.std(obj, axis=(1, 2))])

    filename = "agd_svm_{0}".format(args.dname)
    np.savetxt("{0}_acc.out".format(filename), avg_acc, fmt='%.5f')
    np.savetxt("{0}_obj.out".format(filename), avg_obj, fmt='%.5f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='adaptive sgd')
    parser.add_argument('dname', help='dataset name')
    parser.add_argument('--rep', type=int, default=10)
    parser.add_argument('--delta', type=float, default=1e-8)
    parser.add_argument('--grad_clip', type=float, default=3.0)
    parser.add_argument('--obj_clip', type=float, default=3.0)
    parser.add_argument('--reg_coeff', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--exp_dec', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.1)

    args = parser.parse_args()

    print "Running the program ... [{0}]".format(
        time.strftime("%m/%d/%Y %H:%M:%S"))
    print "Parameters"
    print "----------"

    for arg in vars(args):
        print " - {0:22s}: {1}".format(arg, getattr(args, arg))

    start_time = time.clock()

    main(args)

    elapsed = time.clock() - start_time
    mins, sec = divmod(elapsed, 60)
    hrs, mins = divmod(mins, 60)

    print "The program finished. [{0}]".format(
        time.strftime("%m/%d/%Y %H:%M:%S"))
    print "Elasepd time: %d:%02d:%02d" % (hrs, mins, sec)
