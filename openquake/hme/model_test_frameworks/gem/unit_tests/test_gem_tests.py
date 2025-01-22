import os
import unittest

import numpy as np
import pandas as pd

from openquake.hme.model_test_frameworks.gem.gem_tests import (
    M_test,
    S_test,
    L_test,
    N_test,
    max_mag_check,
    model_mfd_eval,
    rupture_matching_eval,
)


from openquake.hme.utils.tests.load_sm1 import cfg, input_data

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")


class test_gem_tests(unittest.TestCase):
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
            "bad_bins": [],
            "test_data": {
                "obs_loglike": np.array(
                    [-3.68555108, -11.37062494, -1.55414772]
                ),
                "stoch_loglike": np.array(
                    [
                        [
                            -3.68555108,
                            -3.75098033,
                            -5.43856038,
                            -4.35309631,
                            -5.41653736,
                        ],
                        [
                            -9.65914154,
                            -8.2279889,
                            -9.84912031,
                            -7.09742084,
                            -8.6941361,
                        ],
                        [
                            -1.55414772,
                            -2.45676151,
                            -2.45676151,
                            -1.55414772,
                            -6.57898585,
                        ],
                    ]
                ),
                "cell_loglikes": {
                    "836860fffffffff": {
                        "obs_loglike": -3.685551083653414,
                        "stoch_loglikes": np.array(
                            [
                                -3.68555108,
                                -3.75098033,
                                -5.43856038,
                                -4.35309631,
                                -5.41653736,
                            ]
                        ),
                        "bad_bins": [],
                    },
                    "836864fffffffff": {
                        "obs_loglike": -11.370624943255637,
                        "stoch_loglikes": np.array(
                            [
                                -9.65914154,
                                -8.2279889,
                                -9.84912031,
                                -7.09742084,
                                -8.6941361,
                            ]
                        ),
                        "bad_bins": [],
                    },
                    "83694afffffffff": {
                        "obs_loglike": -1.5541477248609592,
                        "stoch_loglikes": np.array(
                            [
                                -1.55414772,
                                -2.45676151,
                                -2.45676151,
                                -1.55414772,
                                -6.57898585,
                            ]
                        ),
                        "bad_bins": [],
                    },
                },
                "cell_fracs": np.array([1.0, 0.0, 1.0]),
            },
        }
        assert S_test_res["critical_pct"] == s_test_res["critical_pct"]
        assert abs(S_test_res["percentile"] - s_test_res["percentile"]) < 0.1
        assert S_test_res["test_pass"] == s_test_res["test_pass"]
        assert S_test_res["test_res"] == s_test_res["test_res"]
        assert S_test_res["bad_bins"] == s_test_res["bad_bins"]
        for resid in np.abs(
            S_test_res["test_data"]["obs_loglike"]
            - s_test_res["test_data"]["obs_loglike"]
        ):
            assert resid < 0.1

    def test_N_test_poisson(self):
        np.random.seed(self.cfg["config"]["rand_seed"])
        N_test_res = N_test(self.cfg, self.input_data)
        n_test_res = {
            "conf_interval_pct": 0.96,
            "conf_interval": (8.0, 24.0),
            "n_pred_earthquakes": 15.581590799999995,
            "n_obs_earthquakes": 14,
            "test_res": "Pass",
            "test_pass": True,
            "test_pass": True,
            "M_min": 6.1,
            "prob_model": "poisson",
        }
        for k, v in N_test_res.items():
            if isinstance(v, float):
                np.testing.assert_approx_equal(v, n_test_res[k])
            else:
                assert v == n_test_res[k]

    def test_M_test(self):
        np.random.seed(self.cfg["config"]["rand_seed"])
        M_test_res = M_test(self.cfg, self.input_data)
        m_test_res = {
            "critical_pct": 0.25,
            "percentile": 0.7,
            "test_pass": True,
            "test_res": "Pass",
            "test_data": {
                "stoch_geom_mean_likes": [
                    0.3921102525278011,
                    0.34589752053184747,
                    0.5230132152251857,
                    0.5480209694183371,
                    0.30084083880346385,
                    0.4603387794170368,
                    0.3717458508138827,
                    0.45001815079620067,
                    0.4152096898871906,
                    0.5496207142742168,
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
            "percentile": 0.4,
            "test_pass": True,
            "test_res": "Pass",
            "bad_bins": [],
            "test_data": {
                "obs_loglike": np.array(
                    [-3.9021022, -11.06163201, -1.72972099]
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
                "obs_loglike_total": -16.693455199476418,
                "stoch_loglike_totals": np.array(
                    [
                        -14.69603894,
                        -15.27304995,
                        -19.58662981,
                        -25.15253911,
                        -15.86097257,
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

    def test_max_mag_check(self):
        np.random.seed(self.cfg["config"]["rand_seed"])
        max_mag_check_res = max_mag_check(self.cfg, self.input_data)
        max_mag_check_results = {
            "test_res": "Pass",
            "test_pass": True,
            "bad_bins": [],
        }
        for k, v in max_mag_check_res.items():
            assert v == max_mag_check_results[k]

    @unittest.skip("removed from test cfg for now")
    def test_model_mfd_eval(self):
        np.random.seed(self.cfg["config"]["rand_seed"])
        Mfd_eval_res = model_mfd_eval(self.cfg, self.input_data)
        mfd_eval_res = {
            "test_data": {
                "mfd_df": pd.DataFrame(
                    data={
                        "bin": np.array(
                            [
                                6.1,
                                6.3,
                                6.5,
                                6.7,
                                6.9,
                                7.1,
                                7.3,
                                7.5,
                                7.7,
                                7.9,
                                8.1,
                                8.3,
                                8.5,
                                8.7,
                            ]
                        ),
                        "mod_mfd": np.array(
                            [
                                0.146072,
                                0.092165,
                                0.058152,
                                0.036692,
                                0.023151,
                                0.014607,
                                0.009217,
                                0.005815,
                                0.003669,
                                0.000000,
                                0.000000,
                                0.000000,
                                0.000000,
                                0.000000,
                            ]
                        ),
                        "mod_mfd_cum": np.array(
                            [
                                0.389540,
                                0.243468,
                                0.151303,
                                0.093150,
                                0.056459,
                                0.033308,
                                0.018701,
                                0.009484,
                                0.003669,
                                0.000000,
                                0.000000,
                                0.000000,
                                0.000000,
                                0.000000,
                            ]
                        ),
                        "obs_mfd": np.array(
                            [
                                0.100,
                                0.125,
                                0.025,
                                0.025,
                                0.025,
                                0.025,
                                0.025,
                                0.000,
                                0.000,
                                0.000,
                                0.000,
                                0.000,
                                0.000,
                                0.000,
                            ]
                        ),
                        "obs_mfd_cum": np.array(
                            [
                                0.350,
                                0.250,
                                0.125,
                                0.100,
                                0.075,
                                0.050,
                                0.025,
                                0.000,
                                0.000,
                                0.000,
                                0.000,
                                0.000,
                                0.000,
                                0.000,
                            ]
                        ),
                    }
                ).set_index("bin")
            }
        }

        for col in Mfd_eval_res["test_data"]["mfd_df"].columns:
            np.testing.assert_allclose(
                Mfd_eval_res["test_data"]["mfd_df"][col].values,
                mfd_eval_res["test_data"]["mfd_df"][col].values,
                atol=1e-4,
            )

    def test_rupture_matching_eval(self):
        rupture_matching_eval_res = rupture_matching_eval(
            self.cfg, self.input_data
        )

        rupture_matching_eval_match_results = pd.read_csv(
            os.path.join(
                TEST_DATA_DIR, "rupture_matching_eval_matched_ruptures.csv"
            ),
            index_col=0,
        )

        pd.testing.assert_frame_equal(
            rupture_matching_eval_res["matched_rups"],
            rupture_matching_eval_match_results,
        )
