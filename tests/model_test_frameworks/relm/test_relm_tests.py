import os
import unittest

import numpy as np

from openquake.hme.core.core import load_inputs
from openquake.hme.model_test_frameworks.relm.relm_tests import S_test, N_test, M_test

BASE_PATH = os.path.dirname(__file__)
SM1_PATH = os.path.join(BASE_PATH, "..", "..", "data", "source_models", "sm1")
DATA_FILE = os.path.join(SM1_PATH, "data", "phl_eqs.csv")


# Doing this here because it takes several seconds and should be done once
cfg = {
    "config": {
        "model_framework": {
            "relm": {
                "N_test": {
                    "prob_model": "poisson",
                    "conf_interval": 0.96,
                    "investigation_time": 40.0,
                },
                "S_test": {
                    "investigation_time": 40.0,
                    "n_iters": 10000,
                    "critical_pct": 0.2,
                    "append": True,
                    "likelihood_fn": "mfd",
                },
            }
        },
        "parallel": False,
        "rand_seed": 69,
    },
    "input": {
        "bins": {
            "mfd_bin_min": 6.5,
            "mfd_bin_max": 8.5,
            "mfd_bin_width": 0.2,
            "h3_res": 3,
        },
        "subset": {"file": None},
        "ssm": {
            "ssm_dir": SM1_PATH + "/",
            "ssm_lt_file": "ssmLT.xml",
            "branch": "b1",
            "tectonic_region_types": ["Active Shallow Crust"],
            "source_types": None,
        },
        "seis_catalog": {
            "seis_catalog_file": DATA_FILE,
            "columns": {
                "time": ["year", "month", "day", "hour", "minute", "second"],
                "source": "Agency",
                "event_id": "eventID",
            },
        },
    },
}

bin_gdf, obs_seis_catalog = load_inputs(cfg)


class test_relm_tests(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg
        self.bin_gdf = bin_gdf
        self.obs_seis_catalog = obs_seis_catalog

    def test_S_test(self):
        np.random.seed(self.cfg["config"]["rand_seed"])
        S_test_res = S_test(self.cfg, self.bin_gdf)
        s_test_res = {
            "critical_pct": 0.2,
            "percentile": 0.22,
            "test_pass": True,
            "test_res": "Pass",
        }
        assert S_test_res["critical_pct"] == s_test_res["critical_pct"]
        assert abs(S_test_res["percentile"] - s_test_res["percentile"]) < 0.1
        assert S_test_res["test_pass"] == s_test_res["test_pass"]
        assert S_test_res["test_res"] == s_test_res["test_res"]

    def test_N_test_poisson(self):
        np.random.seed(self.cfg["config"]["rand_seed"])
        N_test_res = N_test(self.cfg, bin_gdf=self.bin_gdf)
        print(N_test_res)
        rate = 6.0521104000000125

        assert N_test_res["conf_interval_pct"] == 0.96
        assert N_test_res["conf_interval"] == (2.0, 12.0)
        np.testing.assert_almost_equal(N_test_res["inv_time_rate"], rate)
        assert N_test_res["n_obs_earthquakes"] == 3
        assert N_test_res["pass"]
