import unittest

import numpy as np

from openquake.hme.utils import (
    get_cell_rups,
    get_cell_eqs,
    get_mag_bins_from_cfg,
)

from openquake.hme.core.core import cfg_defaults
from openquake.hme.model_test_frameworks.relm.relm_test_functions import (
    mfd_log_likelihood,
    s_test_cell,
    s_test_cells,
    s_test_function,
    N_test_poisson,
)


from openquake.hme.utils.tests.load_sm1 import cfg, input_data


class test_relm_test_functions(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg
        # self.bin_gdf = bin_gdf
        self.rupture_gdf = input_data["rupture_gdf"]
        self.rup_groups = input_data["cell_groups"]
        self.eq_gdf = input_data["eq_gdf"]
        self.eq_groups = input_data["eq_groups"]
        self.s_test_cfg = self.cfg["config"]["model_framework"]["relm"][
            "S_test"
        ]
        self.s_test_cfg["mag_bins"] = get_mag_bins_from_cfg(self.cfg)
        self.s_test_cfg["not_modeled_likelihood"] = 0.0
        self.s_test_cfg["likelihood_fn"] = "mfd"

        self.n_test_cfg = self.cfg["config"]["model_framework"]["relm"][
            "N_test"
        ]

    def test_s_test_cell(self):
        np.random.seed(69)

        annual_rup_rate = self.rupture_gdf.occurrence_rate.sum()
        N_obs = len(self.eq_gdf)
        N_pred = annual_rup_rate * self.s_test_cfg["investigation_time"]
        N_norm = N_obs / N_pred
        self.s_test_cfg["N_norm"] = N_norm

        cell_id = "836860fffffffff"
        cell_rups = get_cell_rups(cell_id, self.rupture_gdf, self.rup_groups)
        cell_eqs = get_cell_eqs(cell_id, self.eq_gdf, self.eq_groups)

        cell_like = s_test_cell(cell_rups, cell_eqs, self.s_test_cfg)

        cell_like_ans = {
            "obs_loglike": -3.685551083653415,
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
        }

        np.testing.assert_almost_equal(
            cell_like["obs_loglike"], cell_like_ans["obs_loglike"]
        )
        np.testing.assert_allclose(
            cell_like["stoch_loglikes"], cell_like_ans["stoch_loglikes"]
        )
        assert cell_like["bad_bins"] == cell_like_ans["bad_bins"]

    def test_s_test_cells_serial(self):
        np.random.seed(69)

        cell_likes = s_test_cells(
            self.rup_groups,
            self.rupture_gdf,
            self.eq_groups,
            self.eq_gdf,
            self.s_test_cfg,
            parallel=False,
        )

        cell_likes_ans = {
            "836860fffffffff": {
                "obs_loglike": -3.685551083653415,
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
        }

        for cell_id, cell_like in cell_likes.items():
            cell_like_ans = cell_likes_ans[cell_id]
            np.testing.assert_almost_equal(
                cell_like["obs_loglike"], cell_like_ans["obs_loglike"]
            )
            np.testing.assert_allclose(
                cell_like["stoch_loglikes"], cell_like_ans["stoch_loglikes"]
            )
            assert cell_like["bad_bins"] == cell_like_ans["bad_bins"]

    def test_s_test_cells_parallel(self):
        np.random.seed(69)

        cell_likes = s_test_cells(
            self.rup_groups,
            self.rupture_gdf,
            self.eq_groups,
            self.eq_gdf,
            self.s_test_cfg,
            parallel=True,
        )

        cell_likes_ans = {
            "836860fffffffff": {
                "obs_loglike": -3.685551083653415,
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
        }

        for cell_id, cell_like in cell_likes.items():
            cell_like_ans = cell_likes_ans[cell_id]
            np.testing.assert_almost_equal(
                cell_like["obs_loglike"], cell_like_ans["obs_loglike"]
            )

            # skipping this because random seed doesn't work with multiprocess
            # np.testing.assert_allclose(
            #    cell_like["stoch_loglikes"], cell_like_ans["stoch_loglikes"]
            # )
            assert cell_like["bad_bins"] == cell_like_ans["bad_bins"]

    def test_s_test_function(self):
        s_test_results = s_test_function(
            self.rupture_gdf,
            self.eq_gdf,
            self.rup_groups,
            self.eq_groups,
            self.s_test_cfg["investigation_time"],
            self.s_test_cfg["n_iters"],
            self.s_test_cfg["likelihood_fn"],
            self.s_test_cfg["mag_bins"],
            self.s_test_cfg["critical_pct"],
            self.s_test_cfg["not_modeled_likelihood"],
            parallel=False,
        )

        s_test_results_ans = {
            "critical_pct": 0.25,
            "percentile": 0.6,
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
                            -4.00885074,
                            -4.70558591,
                            -5.32855537,
                            -7.17313464,
                            -3.87718277,
                        ],
                        [
                            -7.01711583,
                            -10.30585619,
                            -12.8538275,
                            -11.35795879,
                            -7.16168363,
                        ],
                        [
                            -2.84153233,
                            -2.45676151,
                            -5.05921123,
                            -1.55414772,
                            -2.9805893,
                        ],
                    ]
                ),
                "cell_loglikes": {
                    "836860fffffffff": {
                        "obs_loglike": -3.685551083653415,
                        "stoch_loglikes": np.array(
                            [
                                -4.00885074,
                                -4.70558591,
                                -5.32855537,
                                -7.17313464,
                                -3.87718277,
                            ]
                        ),
                        "bad_bins": [],
                    },
                    "836864fffffffff": {
                        "obs_loglike": -11.370624943255637,
                        "stoch_loglikes": np.array(
                            [
                                -7.01711583,
                                -10.30585619,
                                -12.8538275,
                                -11.35795879,
                                -7.16168363,
                            ]
                        ),
                        "bad_bins": [],
                    },
                    "83694afffffffff": {
                        "obs_loglike": -1.5541477248609592,
                        "stoch_loglikes": np.array(
                            [
                                -2.84153233,
                                -2.45676151,
                                -5.05921123,
                                -1.55414772,
                                -2.9805893,
                            ]
                        ),
                        "bad_bins": [],
                    },
                },
                "cell_fracs": np.array([1.0, 0.2, 1.0]),
            },
        }

        test_keys = [
            "critical_pct",
            "percentile",
            "test_pass",
            "test_res",
            "bad_bins",
        ]

        for key in test_keys:
            assert s_test_results[key] == s_test_results_ans[key]

        for cell, cell_data in s_test_results["test_data"][
            "cell_loglikes"
        ].items():
            cell_data_ans = s_test_results_ans["test_data"]["cell_loglikes"][
                cell
            ]
            np.testing.assert_almost_equal(
                cell_data["obs_loglike"], cell_data_ans["obs_loglike"]
            )
            np.testing.assert_almost_equal(
                cell_data["obs_loglike"], cell_data_ans["obs_loglike"]
            )
            np.testing.assert_allclose(
                cell_data["stoch_loglikes"], cell_data_ans["stoch_loglikes"]
            )

    def test_mfd_log_likelihood(self):
        rate_mfd = {
            6.1: 1.5063278628284074,
            6.3: 0.9366653194923473,
            6.5: 0.5819662717863339,
            6.7: 0.3605823827753992,
            6.9: 0.2194124816840237,
            7.1: 0.1270055145570363,
            7.3: 0.06573610995542593,
            7.5: 0.01405758018423675,
            7.7: 0.0,
            7.9: 0.0,
            8.1: 0.0,
        }

        empirical_mfd = {
            6.1: 2.0,
            6.3: 0.0,
            6.5: 0.0,
            6.7: 0.0,
            6.9: 0.0,
            7.1: 0.0,
            7.3: 0.0,
            7.5: 0.0,
            7.7: 0.0,
            7.9: 0.0,
            8.1: 0.0,
        }

        loglikes = [
            -1.3801254232186118,
            -0.9366653194923473,
            -0.5819662717863339,
            -0.3605823827753992,
            -0.2194124816840237,
            -0.1270055145570363,
            -0.06573610995542593,
            -0.01405758018423675,
            0.0,
            0.0,
            0.0,
        ]
        loglike_sum = -3.685551083653415

        ll_sum, lls = mfd_log_likelihood(
            rate_mfd=rate_mfd, empirical_mfd=empirical_mfd, return_likes=True
        )

        np.testing.assert_almost_equal(loglike_sum, ll_sum)
        np.testing.assert_array_almost_equal(np.array(loglikes), np.array(lls))

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
