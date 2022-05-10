import os
import unittest
from copy import deepcopy

import numpy as np

from openquake.hme.utils import deep_update
from openquake.hme.core.core import load_inputs, cfg_defaults
from openquake.hme.model_test_frameworks.relm.relm_test_functions import (
    get_model_mfd,
    get_obs_mfd,
    get_model_annual_eq_rate,
    get_total_obs_eqs,
    subdivide_observed_eqs,
    N_test_poisson,
    N_test_neg_binom,
    mfd_log_likelihood,
    s_test_bin,
)

BASE_PATH = os.path.dirname(__file__)
SM1_PATH = os.path.join(BASE_PATH, "..", "..", "data", "source_models", "sm1")
DATA_FILE = os.path.join(SM1_PATH, "data", "phl_synth_catalog.csv")

# Doing this here because it takes several seconds and should be done once
test_cfg = {
    "meta": {
        "description": "test"},
    "config": {
        "model_framework": {
            "relm": {
                "N_test": {
                    "prob_model": "poisson",
                    "conf_interval": 0.96,
                    "investigation_time": 40.0,
                },
                "M_test": {
                    "prospective": False,
                    "critical_pct": 0.25,
                    "investigation_time": 40.0,
                    "n_iters": 5,
                },
                "S_test": {
                    "prospective": False,
                    "investigation_time": 40.0,
                    "n_iters": 5,
                    "critical_pct": 0.25,
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
            "mfd_bin_min": 6.1,
            "mfd_bin_max": 8.0,
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
            "columns": {"event_id": "event_id"},
        },
        "rupture_file": {
            "rupture_file_path": None,
            "read_rupture_file": False,
            "save_rupture_file": False,
        },
    },
}

cfg = deepcopy(cfg_defaults)
cfg = deep_update(cfg, test_cfg)

bin_gdf, obs_seis_catalog = load_inputs(cfg)


class test_relm_test_functions(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg
        self.bin_gdf = bin_gdf
        self.obs_seis_catalog = obs_seis_catalog

    def test_s_test_bin(self):
        S_test_cfg = self.cfg["config"]["model_framework"]["relm"]["S_test"]
        S_test_cfg["not_modeled_likelihood"] = 0.0

        sb = self.bin_gdf.loc["836860fffffffff"].SpacemagBin

        s_test_bin_res = s_test_bin(sb, S_test_cfg)

    def test_mfd_log_likelihood(self):
        # this test is not specific to N test but this is where t_yrs is stored
        # with this unit test config
        N_test_cfg = self.cfg["config"]["model_framework"]["relm"]["N_test"]
        t_yrs = N_test_cfg["investigation_time"]

        sb = self.bin_gdf.loc["836864fffffffff"].SpacemagBin

        obs_eqs = sb.observed_earthquakes
        rate_mfd = sb.get_rupture_mfd()
        rate_mfd = {mag: t_yrs * rate for mag, rate in rate_mfd.items()}

        mfd_log_like = mfd_log_likelihood(rate_mfd, binned_events=obs_eqs)

        np.testing.assert_almost_equal(mfd_log_like, -11.061632009308141)

    @unittest.skip("not yet implemented")
    def test_subdivide_observed_eqs(self):
        pass

    def test_N_test_poisson(self):
        N_test_cfg = self.cfg["config"]["model_framework"]["relm"]["N_test"]
        n_obs_events = 3
        rate = 0.8645872887605222
        ci = N_test_cfg["conf_interval"]
        N_test_res = N_test_poisson(n_obs_events, rate, ci)

        assert N_test_res["conf_interval_pct"] == 0.96
        assert N_test_res["conf_interval"] == (0.0, 3.0)
        np.testing.assert_almost_equal(N_test_res["inv_time_rate"], rate)
        assert N_test_res["n_obs_earthquakes"] == 3
        assert N_test_res["test_pass"]

    @unittest.skip("not yet implemented")
    def test_N_test_neg_binom(self):
        pass
