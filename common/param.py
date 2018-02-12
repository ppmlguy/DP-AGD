from math import log, sqrt, exp
from scipy.optimize import fsolve


def dp_to_zcdp(eps, delta):
    def eq_epsilon(rho):
        """
        rho-zCDP => (rho + 2*sqrt(rho * ln(1/delta)), delta)-DP
        """
        if rho <= 0.0:
            rhs = rho
        else:
            rhs = rho + 2.0 * sqrt(rho * log(1.0/delta))

        return eps - rhs

    rho = fsolve(eq_epsilon, 0.0)

    return rho[0]


def compute_sigma(eps, delta, sens):
    """
    compute the std. (sigma) for gaussian mechanism
    Parameters
    -----------
    eps, delta : privacy budget
    """
    sigma = sens / eps
    sigma *= sqrt(2.0 * log(1.25 / delta))

    return sigma


def compute_epsilon(rho):
    """
    compute the value of epsilon that satisfies rho-zCDP
    """
    return sqrt(2.0*rho)


def compute_advcomp_budget(eps, delta, T):
    """
    computes the per-iteration privacy budgets using advance composition
    """
    denom = sqrt(2.0 * T * log(2.0 / delta))
    ep = eps / (2.0 * denom)
    dp = delta / (2.0 * T)

    return ep, dp

