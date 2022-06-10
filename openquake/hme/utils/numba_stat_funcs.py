import numpy as np
from numba import jit, njit, prange


@njit
def poisson_sample_scalar_u(mean, u):
    x = 0
    p = np.exp(-mean)
    s = p

    while u > s:
        x += 1
        p *= mean / x
        s += p

    return x


@njit(parallel=True)
def poisson_sample_vec(mean):
    n = len(mean)
    Us = np.random.rand(n)

    n_samples = np.zeros(n, dtype=np.int64)

    for i in prange(n):
        n_samples[i] = poisson_sample_scalar_u(mean[i], Us[i])

    return n_samples


def precompile_poisson_sampling():
    means = np.array([0.1, 0.2])
    samps = poisson_sample_vec(means)


def precompile():
    precompile_poisson_sampling()


precompile()
