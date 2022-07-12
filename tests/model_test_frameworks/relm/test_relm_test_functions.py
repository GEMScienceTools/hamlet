import os
from re import M
import unittest
from copy import deepcopy

import numpy as np

from openquake.hme.utils import (
    deep_update,
    get_cell_rups,
    get_cell_eqs,
    get_mag_bins_from_cfg,
)

from openquake.hme.core.core import load_inputs, cfg_defaults
from openquake.hme.model_test_frameworks.relm.relm_test_functions import (
    s_test_cell,
    s_test_cells,
    s_test_function,
    # get_model_mfd,
    # get_obs_mfd,
    # get_model_annual_eq_rate,
    # get_total_obs_eqs,
    # subdivide_observed_eqs,
    # N_test_poisson,
    # N_test_neg_binom,
    # mfd_log_likelihood,
)

BASE_PATH = os.path.dirname(__file__)
SM1_PATH = os.path.join(BASE_PATH, "..", "..", "data", "source_models", "sm1")
DATA_FILE = os.path.join(SM1_PATH, "data", "phl_synth_catalog.csv")

# Doing this here because it takes several seconds and should be done once
test_cfg = {
    "meta": {"description": "test"},
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

input_data = load_inputs(cfg)


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

    @unittest.skip("not yet implemented")
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

    @unittest.skip("not yet implemented")
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
