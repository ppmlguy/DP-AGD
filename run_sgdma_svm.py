import argparse
import time
import numpy as np
from agd.common.dat import load_dat
from agd.common.svm import svm_loss
from agd.common.svm import svm_test
from agd.common.svm import svm_grad
from sklearn.model_selection import RepeatedKFold
from agd.algo.sgd import dpsgd_ma


def main(args):
    fpath = "./dataset/{0}.dat".format(args.dname)
    X, y = load_dat(fpath, minmax=(0, 1), bias_term=True)
    N, dim = X.shape

    y[y < 1] = -1

    nrep = args.rep
    T = [1, 100, 1000, 10000, 20000]
    epsilon = []
    nT = len(T)
    learning_rate = 0.05
    sigma = 4  # fixed sigma as in MA paper

    K = 5  # 5-folds cross-validation
    cv_rep = 2
    k = 0
    acc = np.zeros((nT, nrep, K*cv_rep))
    obj = np.zeros((nT, nrep, K*cv_rep))

    rkf = RepeatedKFold(n_splits=K, n_repeats=cv_rep)

    for train, test in rkf.split(X):
        train_X, train_y = X[train, :], y[train]
        test_X, test_y = X[test, :], y[test]

        n_train = train_X.shape[0]
        batch_size = int(np.sqrt(n_train) + 10)

        if args.batch_size > 0:
            batch_size = args.batch_size

        for i in range(nT):
            for j in range(nrep):
                sol, eps = dpsgd_ma(train_X, train_y, svm_grad, sigma,
                                    T[i], learning_rate, batch_size,
                                    reg_coeff=args.reg_coeff,
                                    delta=args.delta)
                obj[i, j, k] = svm_loss(sol, train_X, train_y) / n_train
                acc[i, j, k] = svm_test(sol, test_X, test_y) * 100.0
                # print "acc[{},{},{}]={}".format(i, j, k, acc[i, j, k])

                if j == 0 and k == 0:
                    epsilon.append(eps)
        k += 1

    avg_acc = np.vstack([np.array(epsilon),
                         np.mean(acc, axis=(1, 2)),
                         np.std(acc, axis=(1, 2))])
    avg_obj = np.vstack([np.array(epsilon),
                         np.mean(obj, axis=(1, 2)),
                         np.std(obj, axis=(1, 2))])

    filename = "sgdma_svm_{0}".format(args.dname)
    np.savetxt("{0}_acc.out".format(filename), avg_acc, fmt='%.5f')
    np.savetxt("{0}_obj.out".format(filename), avg_obj, fmt='%.5f')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='adaptive sgd')
    parser.add_argument('dname', help='dataset name')
    parser.add_argument('--rep', type=int, default=10)
    parser.add_argument('--reg_coeff', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=-1)
    parser.add_argument('--delta', type=float, default=1e-8)

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
