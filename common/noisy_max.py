import numpy as np


def noisy_max(candidate, lmbda, bmin=False):
    scores = np.array(candidate)
    noise = np.random.exponential(lmbda, size=len(scores))

    # choose the minimum?
    if bmin:
        scores *= -1.0
        noise *= -1.0
    # add noise
    scores += noise
    idx = np.argmax(scores)

    return idx, candidate[idx]


def exp_mech(candidate, sigma, bmin=False, std_noise=None):
    """
    sigma = epsilon/(2*sensitivity)
    """
    scores = np.array(candidate)
    scores *= sigma

    if bmin:
        scores *= -1.0
    else:
        maximum = np.amax(scores)
        scores -= maximum

    prob = np.exp(scores)
    psum = np.sum(prob) * np.random.rand()
    idx = -1

    for i in range(scores.shape[0]):
        psum -= prob[i]
        if psum <= 0.0:
            idx = i
            break

    return idx, candidate[idx]
