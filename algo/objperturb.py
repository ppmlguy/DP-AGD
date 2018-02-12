import numpy as np
from math import log
from math import sqrt
from scipy.optimize import fmin_l_bfgs_b


def out_pert(X, y, eps, f, Lmbda, std_gamma=None, std_gauss=None):
    N, dim = X.shape

    scale = 2./(N * Lmbda * eps)
    # magnitude
    if std_gamma is None:
        mag = np.random.gamma(dim, scale=scale)
    else:
        mag = scale * std_gamma

    # sampling from the surface of unit sphere
    if std_gauss is None:
        rdir = np.random.randn(dim)
    else:
        rdir = std_gauss
    rdir /= np.linalg.norm(rdir, 2)

    b = mag * rdir

    x0 = np.zeros(dim)
    x, fmin, info = fmin_l_bfgs_b(f, x0, fprime=None,
                                  args=(X, y, Lmbda), maxiter=500)

    return x + b


def obj_pert(X, y, eps, delta, f, zeta, lmbda, reg_coeff=0.0, std_noise=None):
    """ object perturbation method.
    Parameters
    ----------
    X: training set, shape=(n_samples, n_features)
    zeta: || \grad \ell(\theta) ||_2
    lmbda: upper bound on the eigenvalues of hessian
    """
    n_samples, n_features = X.shape

    Delta = (2.0 * lmbda) / eps

    if delta == 0.0:
        magnitude = np.random.gamma(n_features, scale=(2.*zeta)/eps)
        direction = np.random.randn(n_features)
        direction /= np.linalg.norm(direction, 2)
        b = magnitude * direction
    else:
        sigma = 8.0 * log(2.0/delta) + 4.0 * eps
        sigma *= ((zeta/eps)**2)
        sigma = sqrt(sigma)

        if std_noise is None:
            std_noise = np.random.randn(n_features)
        b = std_noise * sigma

    x0 = np.zeros(n_features)

    x, fmin, info = fmin_l_bfgs_b(perturbed_objfunc, x0, fprime=None,
                                  args=(X, y, b, Delta, f, reg_coeff),
                                  maxiter=300)

    if info['warnflag'] > 0:
        print "(obj_pert) Failed to converge: ", info['task']
        print "niter=", info['nit']

    return x


def perturbed_objfunc(w, *args):
    X = args[0]
    y = args[1]
    b = args[2]
    Delta = args[3]
    loss_and_grad = args[4]
    reg_coeff = args[5]

    N, dim = X.shape
    n = 1.0 * N

    c1 = Delta / n

    objval, grad = loss_and_grad(w, X, y, reg_coeff=reg_coeff)

    wp = np.copy(w)
    wp[0] = 0

    objval += np.dot(wp, wp) / n
    objval += 0.5 * c1 * np.dot(wp, wp)
    objval += np.dot(b, w) / n

    grad += (2. / n) * wp
    grad += c1 * wp
    grad += b / n

    return objval, grad
