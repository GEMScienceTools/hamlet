import unittest

import numpy as np

from openquake.hme.model_test_frameworks.relm.relm_tests import (
    S_test,
    N_test,
    M_test,
    L_test,
)

from openquake.hme.utils.tests.load_sm1 import cfg, input_data


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
            "percentile": 0.8,
            "test_pass": True,
            "test_res": "Pass",
            "test_data": {
                "obs_loglike": np.array(
                    [-3.81175352, -8.63409875, -1.55414772]
                ),
            },
        }
        assert S_test_res["critical_pct"] == s_test_res["critical_pct"]
        assert abs(S_test_res["percentile"] - s_test_res["percentile"]) < 0.1
        assert S_test_res["test_pass"] == s_test_res["test_pass"]
        assert S_test_res["test_res"] == s_test_res["test_res"]
        for resid in np.abs(
            S_test_res["test_data"]["obs_loglike"]
            - s_test_res["test_data"]["obs_loglike"]
        ):
            assert resid < 0.1

    def test_N_test_poisson(self):
        np.random.seed(self.cfg["config"]["rand_seed"])
        N_test_res = N_test(self.cfg, self.input_data)
        rate = 15.581590799999995
        assert N_test_res["conf_interval_pct"] == 0.96
        assert N_test_res["conf_interval"] == (8.0, 24.0)
        np.testing.assert_almost_equal(N_test_res["inv_time_rate"], rate)
        assert N_test_res["n_obs_earthquakes"] == 14
        assert N_test_res["test_pass"]

    def test_M_test(self):
        np.random.seed(self.cfg["config"]["rand_seed"])
        M_test_res = M_test(self.cfg, self.input_data)
        m_test_res = {
            "critical_pct": 0.25,
            "percentile": 0.3,
            "test_pass": True,
            "test_res": "Pass",
            "test_data": {
                "stoch_geom_mean_likes": [
                    0.40497733396760993,
                    0.3655365204749642,
                    0.5081278905683008,
                    0.5406274396105439,
                    0.30834651297558485,
                    0.44723721336802896,
                    0.3723810523509699,
                    0.44394680900887606,
                    0.43544222084572126,
                    0.5299113170722966,
                ],
                "obs_geom_mean_like": 0.4775592008485948,
            },
        }

        assert M_test_res["critical_pct"] == m_test_res["critical_pct"]
        assert M_test_res["percentile"] == m_test_res["percentile"]
        assert M_test_res["test_pass"] == m_test_res["test_pass"]
        assert M_test_res["test_res"] == m_test_res["test_res"]
        assert (
            M_test_res["test_data"]["obs_geom_mean_like"]
            == m_test_res["test_data"]["obs_geom_mean_like"]
        )
        for i, ll in enumerate(
            M_test_res["test_data"]["stoch_geom_mean_likes"]
        ):
            np.testing.assert_almost_equal(
                ll, m_test_res["test_data"]["stoch_geom_mean_likes"][i]
            )

    def test_L_test(self):
        np.random.seed(self.cfg["config"]["rand_seed"])
        L_test_res = L_test(self.cfg, self.input_data)
        l_test_res = {
            "critical_pct": 0.25,
            "percentile": 0.6,
            "test_pass": True,
            "test_res": "Pass",
            "bad_bins": [],
            "test_data": {
                "obs_loglike": np.array(
                    [-4.24237026, -9.60949955, -1.72972099]
                ),
                "stoch_loglike": np.array(
                    [
                        [-3.9021022, -8.78647774, -2.007459],
                        [-3.86049863, -8.43420713, -2.97834419],
                        [-5.54807868, -8.7763415, -5.26220963],
                        [-5.58968225, -9.9454311, -9.61742576],
                        [-5.31199004, -8.81926154, -1.72972099],
                    ]
                ),
            },
        }
        assert L_test_res["critical_pct"] == l_test_res["critical_pct"]
        assert L_test_res["percentile"] == l_test_res["percentile"]
        assert L_test_res["test_pass"] == l_test_res["test_pass"]
        assert L_test_res["test_res"] == l_test_res["test_res"]
        for i, ll in enumerate(L_test_res["test_data"]["obs_loglike"]):
            np.testing.assert_almost_equal(
                ll, l_test_res["test_data"]["obs_loglike"][i]
            )
        np.testing.assert_array_almost_equal(
            L_test_res["test_data"]["stoch_loglike"],
            l_test_res["test_data"]["stoch_loglike"],
        )
