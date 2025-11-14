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
            "critical_frac": 0.25,
            "fractile": 0.4,
            "test_pass": True,
            "test_res": "Pass",
            "bad_bins": [],
            "unmatched_eqs": [],
            "test_data": {
                "obs_loglike": np.array(
                    [-3.9021022, -11.06163201, -1.72972099]
                ),
                "stoch_loglike": np.array(
                    [
                        [
                            -3.9021022,
                            -3.86049863,
                            -5.54807868,
                            -5.58968225,
                            -5.31199004,
                        ],
                        [
                            -8.78647774,
                            -8.43420713,
                            -8.7763415,
                            -9.9454311,
                            -8.81926154,
                        ],
                        [
                            -2.007459,
                            -2.97834419,
                            -5.26220963,
                            -9.61742576,
                            -1.72972099,
                        ],
                    ]
                ),
                "cell_loglikes": {
                    "836860fffffffff": {
                        "obs_loglike": -3.9021021979158066,
                        "stoch_loglikes": np.array(
                            [
                                -3.9021022,
                                -3.86049863,
                                -5.54807868,
                                -5.58968225,
                                -5.31199004,
                            ]
                        ),
                        "bad_bins": [],
                        "unmatched_eqs": [],
                        "obs_rate": 2,
                        "mod_rate": 4.24237025928183,
                        "obs_mfd": {
                            6.1: 2,
                            6.3: 0,
                            6.5: 0,
                            6.7: 0,
                            6.9: 0,
                            7.1: 0,
                            7.3: 0,
                            7.5: 0,
                            7.7: 0,
                            7.9: 0,
                            8.1: 0,
                            8.3: 0,
                            8.5: 0,
                            8.7: 0,
                        },
                        "mod_mfd": {
                            6.1: 1.6764988835164836,
                            6.3: 1.042481123205787,
                            6.5: 0.6477114504554455,
                            6.7: 0.40131765272108844,
                            6.9: 0.2441996790009251,
                            7.1: 0.14135342551222732,
                            7.3: 0.07316236900780378,
                            7.5: 0.01564567586206897,
                            7.7: 0.0,
                            7.9: 0.0,
                            8.1: 0.0,
                            8.3: 0.0,
                            8.5: 0.0,
                            8.7: 0.0,
                        },
                    },
                    "836864fffffffff": {
                        "obs_loglike": -11.061632009308152,
                        "stoch_loglikes": np.array(
                            [
                                -8.78647774,
                                -8.43420713,
                                -8.7763415,
                                -9.9454311,
                                -8.81926154,
                            ]
                        ),
                        "bad_bins": [],
                        "unmatched_eqs": [],
                        "obs_rate": 12,
                        "mod_rate": 9.60949954846571,
                        "obs_mfd": {
                            6.1: 2,
                            6.3: 5,
                            6.5: 1,
                            6.7: 1,
                            6.9: 1,
                            7.1: 1,
                            7.3: 1,
                            7.5: 0,
                            7.7: 0,
                            7.9: 0,
                            8.1: 0,
                            8.3: 0,
                            8.5: 0,
                            8.7: 0,
                        },
                        "mod_mfd": {
                            6.1: 3.408881063150183,
                            6.3: 2.1928051212259665,
                            6.5: 1.4110856599207922,
                            6.7: 0.9219710447278912,
                            6.9: 0.6067196445883443,
                            7.1: 0.4088094597224057,
                            7.3: 0.2954980309921962,
                            7.5: 0.21696312413793106,
                            7.7: 0.14676640000000002,
                            7.9: 0.0,
                            8.1: 0.0,
                            8.3: 0.0,
                            8.5: 0.0,
                            8.7: 0.0,
                        },
                    },
                    "83694afffffffff": {
                        "obs_loglike": -1.7297209922524603,
                        "stoch_loglikes": np.array(
                            [
                                -2.007459,
                                -2.97834419,
                                -5.26220963,
                                -9.61742576,
                                -1.72972099,
                            ]
                        ),
                        "bad_bins": [],
                        "unmatched_eqs": [],
                        "obs_rate": 0,
                        "mod_rate": 1.7297209922524603,
                        "obs_mfd": {
                            6.1: 0,
                            6.3: 0,
                            6.5: 0,
                            6.7: 0,
                            6.9: 0,
                            7.1: 0,
                            7.3: 0,
                            7.5: 0,
                            7.7: 0,
                            7.9: 0,
                            8.1: 0,
                            8.3: 0,
                            8.5: 0,
                            8.7: 0,
                        },
                        "mod_mfd": {
                            6.1: 0.7574952533333335,
                            6.3: 0.45131895556824664,
                            6.5: 0.26729328962376236,
                            6.7: 0.14437490255102037,
                            6.9: 0.07511387641073079,
                            7.1: 0.03412471476536682,
                            7.3: 0.0,
                            7.5: 0.0,
                            7.7: 0.0,
                            7.9: 0.0,
                            8.1: 0.0,
                            8.3: 0.0,
                            8.5: 0.0,
                            8.7: 0.0,
                        },
                    },
                },
                "cell_fracs": np.array([0.8, 0.0, 1.0]),
            },
        }
        assert S_test_res["critical_frac"] == s_test_res["critical_frac"]
        assert abs(S_test_res["fractile"] - s_test_res["fractile"]) < 0.1
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
        assert N_test_res["conf_interval_frac"] == 0.96
        assert N_test_res["conf_interval"] == (8.0, 24.0)
        np.testing.assert_almost_equal(N_test_res["n_pred_earthquakes"], rate)
        assert N_test_res["n_obs_earthquakes"] == 14
        assert N_test_res["test_pass"]

    def test_M_test(self):
        np.random.seed(self.cfg["config"]["rand_seed"])
        M_test_res = M_test(self.cfg, self.input_data)
        m_test_res = {
            "critical_frac": 0.25,
            "fractile": 0.6,
            "test_pass": True,
            "test_res": "Pass",
            "test_data": {
                "stoch_geom_mean_likes": [
                    0.5447108990283018,
                    0.4679482957482628,
                    0.5462402491309363,
                    0.49791781575129235,
                    0.43913424211435714,
                    0.4173641310254096,
                    0.4786239205533635,
                    0.4154899302905408,
                    0.4366305278303349,
                    0.40092171290704126,
                ],
                "obs_geom_mean_like": 0.4775592008485948,
                "stochastic_eq_counts": {
                    6.1: [6, 5, 5, 5, 8, 4, 4, 2, 2, 10],
                    6.3: [4, 5, 3, 2, 4, 1, 1, 1, 5, 4],
                    6.5: [3, 1, 2, 2, 4, 3, 2, 1, 3, 3],
                    6.7: [1, 2, 1, 1, 0, 3, 1, 1, 2, 2],
                    6.9: [1, 1, 1, 0, 0, 3, 0, 0, 2, 2],
                    7.1: [0, 1, 1, 0, 0, 0, 0, 0, 1, 0],
                    7.3: [0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
                    7.5: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    7.7: [0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
                    7.9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    8.1: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    8.3: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    8.5: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    8.7: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                },
                "model_mfd": {
                    6.1: 5.8428752,
                    6.3: 3.686605199999999,
                    6.5: 2.3260904,
                    6.7: 1.4676635999999998,
                    6.9: 0.9260332,
                    7.1: 0.5842875999999999,
                    7.3: 0.36866039999999994,
                    7.5: 0.23260880000000006,
                    7.7: 0.14676640000000002,
                    7.9: 0.0,
                    8.1: 0.0,
                    8.3: 0.0,
                    8.5: 0.0,
                    8.7: 0.0,
                },
                "model_mfd_norm": {
                    6.1: 5.2498011178678885,
                    6.3: 3.312400733819809,
                    6.5: 2.0899833667817798,
                    6.7: 1.3186901558215738,
                    6.9: 0.8320373039189298,
                    7.1: 0.5249801836664841,
                    7.3: 0.33123996556243795,
                    7.5: 0.20899812103909193,
                    7.7: 0.1318690515220051,
                    7.9: 0.0,
                    8.1: 0.0,
                    8.3: 0.0,
                    8.5: 0.0,
                    8.7: 0.0,
                },
                "obs_mfd": {
                    6.1: 4.0,
                    6.3: 5.0,
                    6.5: 1.0,
                    6.7: 1.0,
                    6.9: 1.0,
                    7.1: 1.0,
                    7.3: 1.0,
                    7.5: 0.0,
                    7.7: 0.0,
                    7.9: 0.0,
                    8.1: 0.0,
                    8.3: 0.0,
                    8.5: 0.0,
                    8.7: 0.0,
                },
            },
        }

        assert M_test_res["critical_frac"] == m_test_res["critical_frac"]
        assert M_test_res["fractile"] == m_test_res["fractile"]
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
            "critical_frac": 0.25,
            "fractile": 0.4,
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
        assert L_test_res["critical_frac"] == l_test_res["critical_frac"]
        assert L_test_res["fractile"] == l_test_res["fractile"]
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
