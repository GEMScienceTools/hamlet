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
                    "n_iters": 100,
                    "critical_pct": 0.25,
                    "append": True,
                },
            }
        },
        "parallel": False,
        "rand_seed": 69,
    },
    "input": {
        "bins": {"mfd_bin_min": 6.5, "mfd_bin_max": 8.5, "mfd_bin_width": 0.2},
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
        S_test_res = S_test(self.cfg, self.bin_gdf)
        s_test_res = {
            "critical_pct": 0.25,
            "percentile": 0.01,
            "test_pass": False,
            "test_res": "Fail",
        }
        assert S_test_res == s_test_res

    def test_N_test_poisson(self):
        N_test_res = N_test(self.cfg, bin_gdf=self.bin_gdf)
        rate = 0.8645872887605222

        assert N_test_res["conf_interval_pct"] == 0.96
        assert N_test_res["conf_interval"] == (0.0, 3.0)
        np.testing.assert_almost_equal(N_test_res["inv_time_rate"], rate)
        assert N_test_res["n_obs_earthquakes"] == 3
        assert N_test_res["pass"]
