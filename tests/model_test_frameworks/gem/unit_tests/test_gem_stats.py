import unittest
import os

import numpy as np
from scipy.stats import poisson
from shapely.geometry import Polygon

from openquake import hme
from openquake.hme.utils import SimpleRupture, Earthquake
from openquake.hme.utils.bins import MagBin, SpacemagBin

from openquake.hme.model_test_frameworks.gem.gem_stats import (
    calc_mag_bin_empirical_likelihood,
    calc_mag_bin_likelihood, 
    calc_mag_bin_poisson_likelihood,
    calc_mfd_log_likelihood_independent)


def test_calc_mag_bin_empirical_likelihood_events():
    rate_dict = {0: 9.80198673e-01, 
                 1: 1.96039735e-02, 
                 2: 1.96039735e-04,
                 3: 1.30693156e-06,
                 4: 6.53465782e-09}

    assert calc_mag_bin_empirical_likelihood(0, rate_dict) == 9.80198673e-01
    assert calc_mag_bin_empirical_likelihood(1, rate_dict) == 1.96039735e-02 
    assert calc_mag_bin_empirical_likelihood(2, rate_dict) == 1.96039735e-04
    assert calc_mag_bin_empirical_likelihood(3, rate_dict) == 1.30693156e-06
    assert calc_mag_bin_empirical_likelihood(4, rate_dict) == 6.53465782e-09
    


def test_calc_mag_bin_empirical_likelihood_no_events():
    rate_dict = {0: 0.99, 1: 0.01}

    assert calc_mag_bin_empirical_likelihood(2, 
        rate_dict, not_modeled_val=1e-5) == 1e-5


def test_calc_mag_bin_poisson_likelihood_events():
    rate = 0.02

    n_events = [0, 1, 2, 3, 4]

    probs = np.array([9.80198673e-01, 1.96039735e-02, 1.96039735e-04, 
                      1.30693156e-06, 6.53465782e-09])

    my_probs = np.array([calc_mag_bin_poisson_likelihood(n, rate) 
        for n in n_events])

    np.testing.assert_allclose(probs, my_probs)


def test_calc_mag_bin_poisson_likelihood_no_events():
    prob_no_events = calc_mag_bin_poisson_likelihood(1, 0., 
        not_modeled_val=1e-5)
    
    assert prob_no_events == 1e-5


def test_calc_mag_bin_likelihood_w_poisson():
    prob = calc_mag_bin_likelihood(1, 0.02, time_interval=1., 
                                   likelihood_method='poisson')
    my_prob = 1.96039735e-02
    
    np.testing.assert_allclose(prob, my_prob)


class TestGEMLikelihoodFunctions(unittest.TestCase):
    def setUp(self):
        pass