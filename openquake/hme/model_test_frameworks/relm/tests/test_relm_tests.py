import os
import unittest
from copy import deepcopy

import numpy as np

from openquake.hme.utils import deep_update
from openquake.hme.core.core import load_inputs, cfg_defaults
from openquake.hme.model_test_frameworks.relm.relm_tests import (
    S_test,
    N_test,
    M_test,
)


from openquake.hme.utils.tests.load_sm1 import cfg, input_data, eq_gdf, rup_gdf


class test_relm_tests(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg
        self.input_data = input_data
        self.rupture_gdf = input_data["rupture_gdf"]
        self.rup_groups = input_data["cell_groups"]
        self.eq_gdf = input_data["eq_gdf"]
        self.eq_groups = input_data["eq_groups"]

    def test_S_test(self):
        np.random.seed(self.cfg["config"]["rand_seed"])
        S_test_res = S_test(self.cfg, self.input_data)
        s_test_res = {
            "critical_pct": 0.25,
            "percentile": 0.4,
            "test_pass": True,
            "test_res": "Pass",
            "test_data": {
            "obs_loglike": np.array([ -3.68555108, -11.37062494,  -1.55414772]),
            },
        }
        assert S_test_res["critical_pct"] == s_test_res["critical_pct"]
        assert abs(S_test_res["percentile"] - s_test_res["percentile"]) < 0.1
        assert S_test_res["test_pass"] == s_test_res["test_pass"]
        assert S_test_res["test_res"] == s_test_res["test_res"]
        for resid in np.abs(S_test_res["test_data"]["obs_loglike"] - 
                   s_test_res["test_data"]["obs_loglike"]):
            assert resid < 0.1

    def test_N_test_poisson(self):
        np.random.seed(self.cfg["config"]["rand_seed"])
        N_test_res = N_test(self.cfg, self.input_data)
        print(N_test_res)
        rate = 15.581590799999995

        assert N_test_res["conf_interval_pct"] == 0.96
        assert N_test_res["conf_interval"] == (8.0, 24.0)
        np.testing.assert_almost_equal(N_test_res["inv_time_rate"], rate)
        assert N_test_res["n_obs_earthquakes"] == 14
        assert N_test_res["test_pass"]
