import argparse
import time
import numpy as np
from agd.common.dat import load_dat
from agd.common.svm import svm_loss
from agd.common.svm import svm_test
from agd.common.svm import svm_grad
from sklearn.model_selection import RepeatedKFold
from agd.algo.sgd import dpsgd_adv


def main(args):
    fpath = "./dataset/{0}.dat".format(args.dname)
    X, y = load_dat(fpath, minmax=(0, 1), bias_term=True)
    y[y < 1] = -1
    N, dim = X.shape

    nrep = args.rep
    epsilon = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    neps = len(epsilon)
    learning_rate = 0.1

    K = 5  # 5-folds cross-validation
    cv_rep = 3
    k = 0
    acc = np.zeros((neps, nrep, K*cv_rep))
    obj = np.zeros((neps, nrep, K*cv_rep))

    rkf = RepeatedKFold(n_splits=K, n_repeats=cv_rep)

    for train, test in rkf.split(X):
        train_X, train_y = X[train, :], y[train]
        test_X, test_y = X[test, :], y[test]

        n_train = train_X.shape[0]
        batch_size = int(np.sqrt(n_train) + 10)

        if args.batch_size > 0:
            batch_size = args.batch_size

        for i, eps in enumerate(epsilon):
            # number of iterations
            T = max(int(round((n_train * eps) / 500.0)), 1)
            if args.T > 0:
                T = int(args.T * eps)

            for j in range(nrep):
                sol = dpsgd_adv(train_X, train_y, svm_grad, eps, T,
                                learning_rate, batch_size,
                                reg_coeff=args.reg_coeff)
                obj[i, j, k] = svm_loss(sol, train_X, train_y) / n_train
                acc[i, j, k] = svm_test(sol, test_X, test_y) * 100.0

        k += 1

    avg_acc = np.vstack([np.mean(acc, axis=(1, 2)),
                         np.std(acc, axis=(1, 2))])
    avg_obj = np.vstack([np.mean(obj, axis=(1, 2)),
                         np.std(obj, axis=(1, 2))])

    filename = "sgdadv_svm_{0}".format(args.dname)
    np.savetxt("{0}_acc.out".format(filename), avg_acc, fmt='%.5f')
    np.savetxt("{0}_obj.out".format(filename), avg_obj, fmt='%.5f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='adaptive sgd')
    parser.add_argument('dname', help='dataset name')
    parser.add_argument('--rep', type=int, default=10)
    parser.add_argument('--reg_coeff', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--T', type=int, default=-1)

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
