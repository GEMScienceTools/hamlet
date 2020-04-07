import os
import unittest

import numpy as np

from openquake.hme.core.core import load_inputs
from openquake.hme.model_test_frameworks.relm.relm_test_functions import (
    get_model_mfd,
    get_obs_mfd,
    get_model_annual_eq_rate,
    get_total_obs_eqs,
    subdivide_observed_eqs,
    N_test_poisson,
    N_test_neg_binom,
    mfd_log_likelihood,
)

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
                }
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


class test_relm_test_functions(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg
        self.bin_gdf = bin_gdf
        self.obs_seis_catalog = obs_seis_catalog

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

        np.testing.assert_almost_equal(mfd_log_like, -8.213565944294217)

    def test_get_model_mfd_noncum(self):
        mod_mfd = get_model_mfd(self.bin_gdf)

        mod_mfd_res = {
            6.5: 0.008307465898659848,
            6.7: 0.005241656622778132,
            6.9: 0.003307261743385139,
            7.1: 0.0020867410871072255,
            7.3: 0.0013166446149388937,
            7.5: 0.0008307465898659899,
            7.7: 0.0005241656622778221,
            7.9: 0,
            8.1: 0,
            8.3: 0,
            8.5: 0,
            8.7: 0,
        }

        for k, v in mod_mfd.items():
            np.testing.assert_almost_equal(v, mod_mfd_res[k])

    def test_get_model_mfd_cum(self):
        mod_mfd = get_model_mfd(self.bin_gdf, cumulative=True)

        mod_mfd_res = {
            6.5: 0.02161468221901305,
            6.7: 0.013307216320353202,
            6.9: 0.00806555969757507,
            7.1: 0.004758297954189932,
            7.3: 0.002671556867082706,
            7.5: 0.0013549122521438121,
            7.7: 0.0005241656622778221,
            7.9: 0.0,
            8.1: 0.0,
            8.3: 0.0,
            8.5: 0.0,
            8.7: 0.0,
        }

        for k, v in mod_mfd.items():
            np.testing.assert_almost_equal(v, mod_mfd_res[k])

    def test_get_obs_mfd_noncum(self):
        N_test_cfg = self.cfg["config"]["model_framework"]["relm"]["N_test"]
        t_yrs = N_test_cfg["investigation_time"]
        obs_mfd = get_obs_mfd(self.bin_gdf, t_yrs=t_yrs, cumulative=False)
        obs_mfd_res = {
            6.5: 0.025,
            6.7: 0.0,
            6.9: 0.0,
            7.1: 0.0,
            7.3: 0.0,
            7.5: 0.05,
            7.7: 0.0,
            7.9: 0.0,
            8.1: 0.0,
            8.3: 0.0,
            8.5: 0.0,
            8.7: 0.0,
        }

        for k, v in obs_mfd.items():
            np.testing.assert_almost_equal(v, obs_mfd_res[k])

    def test_get_obs_mfd_cum(self):
        N_test_cfg = self.cfg["config"]["model_framework"]["relm"]["N_test"]
        t_yrs = N_test_cfg["investigation_time"]
        obs_mfd = get_obs_mfd(self.bin_gdf, t_yrs=t_yrs, cumulative=True)
        obs_mfd_res = {
            6.5: 0.075,
            6.7: 0.05,
            6.9: 0.05,
            7.1: 0.05,
            7.3: 0.05,
            7.5: 0.05,
            7.7: 0.0,
            7.9: 0.0,
            8.1: 0.0,
            8.3: 0.0,
            8.5: 0.0,
            8.7: 0.0,
        }

        for k, v in obs_mfd.items():
            np.testing.assert_almost_equal(v, obs_mfd_res[k])

    def test_get_model_annual_eq_rate(self):
        ann_eq_rate = get_model_annual_eq_rate(self.bin_gdf)
        rate = 0.02161468221901305
        np.testing.assert_almost_equal(ann_eq_rate, rate)

    def test_get_total_obs_eqs(self):
        obs_eqs = get_total_obs_eqs(self.bin_gdf)
        # just test some aspects instead of instantiating Earthquakes
        for eq in obs_eqs:
            assert eq.magnitude in [6.44, 7.4, 7.54]
        assert len(obs_eqs) == 3

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
        assert N_test_res["pass"]

    @unittest.skip("not yet implemented")
    def test_N_test_neg_binom(self):
        pass
