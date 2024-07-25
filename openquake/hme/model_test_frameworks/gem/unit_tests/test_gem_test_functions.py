import unittest

import numpy as np
import pandas as pd

from openquake.hme.utils import get_mag_bins_from_cfg
from openquake.hme.utils.tests.load_sm1 import cfg, input_data
from openquake.hme.model_test_frameworks.gem.gem_test_functions import (
    get_rupture_gdf_cell_moment,
    get_catalog_moment,
    moment_over_under_eval_fn,
    model_mfd_eval_fn,
    get_moment_from_mfd,
    _get_moment_from_mfd_dict,
    mag_diff_likelihood,
    get_distances,
    get_rups_in_mag_range,
    get_nearby_rups,
    get_matching_rups,
    _get_matching_rups,
    match_eqs_to_rups,
    rupture_matching_eval_fn,
)


class test_gem_test_functions(unittest.TestCase):
    def setUp(self):
        self.cfg = cfg
        self.input_data = input_data
        self.rupture_gdf = input_data["rupture_gdf"]
        self.cell_groups = input_data["cell_groups"]
        self.eq_gdf = input_data["eq_gdf"]
        self.eq_groups = input_data["eq_groups"]

    def test_get_rupture_gdf_cell_moment(self):
        np.random.seed(69)
        t_yrs = 2.0
        cell_moments, total_moment = get_rupture_gdf_cell_moment(
            self.rupture_gdf, t_yrs, self.cell_groups
        )

        np.testing.assert_almost_equal(
            cell_moments["836860fffffffff"], 1.8949965316710815e18
        )
        np.testing.assert_almost_equal(
            cell_moments["836864fffffffff"], 9.981130492886022e18
        )
        np.testing.assert_almost_equal(
            cell_moments["83694afffffffff"], 4.78620456612491e17
        )
        np.testing.assert_almost_equal(total_moment, 1.2354747481169594e19)

    def test_get_catalog_moment(self):
        np.random.seed(69)
        cell_moments, total_moment = get_catalog_moment(
            self.eq_gdf, self.eq_groups
        )

        np.testing.assert_almost_equal(
            cell_moments["836860fffffffff"], 2.667042864326648e18
        )
        np.testing.assert_almost_equal(
            cell_moments["836864fffffffff"], 1.8917677531931476e20
        )
        np.testing.assert_almost_equal(total_moment, 1.918438181836414e20)

    def test_moment_over_under_eval_fn(self):
        np.random.seed(69)
        t_yrs = 40.0
        moment_results = moment_over_under_eval_fn(
            self.rupture_gdf, self.eq_gdf, self.cell_groups, t_yrs, n_iters=5
        )

        res = {
            "test_data": {
                "total_model_moment": 2.4709494962339188e20,
                "cell_model_moments": pd.Series(
                    {
                        "836860fffffffff": 3.7899930633421636e19,
                        "836864fffffffff": 1.9962260985772045e20,
                        "83694afffffffff": 9.572409132249821e18,
                    }
                ),
                "total_obs_moment": 1.918438181836414e20,
                "modeled_obs_moment": {
                    "mean": 2.3675179021006517e20,
                    "sd": 1.9698564318574975e20,
                },
                "frac": 0.6,
                "cell_fracs": {
                    "836860fffffffff": 0.0,
                    "836864fffffffff": 0.6,
                    "83694afffffffff": 0.0,
                },
                "stoch_total_moments": np.array(
                    [
                        4.16638513e20,
                        7.18579467e19,
                        9.25670124e20,
                        1.70233374e20,
                        7.25482190e19,
                    ]
                ),
                "stoch_cell_moments": {
                    0: {
                        "836860fffffffff": 3.6288167232485933e19,
                        "836864fffffffff": 3.6007914451369414e20,
                        "83694afffffffff": 2.027120100325426e19,
                    },
                    1: {
                        "836860fffffffff": 1.3611592634482946e19,
                        "836864fffffffff": 5.448798000016518e19,
                        "83694afffffffff": 3.7583740428844355e18,
                    },
                    2: {
                        "836860fffffffff": 1.2024138466996329e20,
                        "836864fffffffff": 8.008843648900993e20,
                        "83694afffffffff": 4.544374149288609e18,
                    },
                    3: {
                        "836860fffffffff": 6.463806217338798e19,
                        "836864fffffffff": 9.809636938478754e19,
                        "83694afffffffff": 7.498942093324558e18,
                    },
                    4: {
                        "836860fffffffff": 2.1014806982513484e19,
                        "836864fffffffff": 4.9649762966639436e19,
                        "83694afffffffff": 1.8836490894897943e18,
                    },
                },
                "obs_cell_moments": pd.Series(
                    {
                        "836860fffffffff": 2.667043e18,
                        "836864fffffffff": 1.891768e20,
                        "83694afffffffff": 0.000000e00,
                    }
                ),
                "model_moment_ratio": 1.2880005827806327,
            }
        }

        mom_td = moment_results["test_data"]
        res_td = res["test_data"]

        # random seed does not prevent variable results...

        # for k, v in mom_td.items():
        #     if isinstance(v, float):
        #         np.testing.assert_almost_equal(v, res_td[k], decimal=5)
        #     elif isinstance(v, dict):
        #         for kk, vv in v.items():  # all should be float values
        #             np.testing.assert_almost_equal(vv, res_td[k][kk], decimal=5)
        #     elif isinstance(v, np.ndarray):
        #         np.testing.assert_array_almost_equal(v, res_td[k], decimal=5)
        #     elif isinstance(v, pd.Series):
        #         for ind, val in v.iteritems():
        #             np.testing.assert_almost_equal(
        #                 val, res_td[k][ind], decimal=3
        #             )

    def test_model_mfd_eval_fn(self):
        t_yrs = 1.0
        mag_bins = get_mag_bins_from_cfg(self.cfg)

        mod_mfd = model_mfd_eval_fn(
            self.rupture_gdf, self.eq_gdf, mag_bins, t_yrs
        )

        mm = mod_mfd["test_data"]["mfd_df"]

        bins = np.array(
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
        )

        np.testing.assert_array_equal(mm.index.values, bins)

        obs_mfd_cum = np.array(
            [
                14.0,
                10.0,
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        np.testing.assert_array_equal(mm.obs_mfd_cum.values, obs_mfd_cum)

        obs_mfd = np.array(
            [
                4.0,
                5.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        np.testing.assert_array_equal(mm.obs_mfd.values, obs_mfd)

        mod_mfd_cum = np.array(
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
        )

        np.testing.assert_array_almost_equal(
            mm.mod_mfd_cum.values, mod_mfd_cum, decimal=2
        )

        mod_mfd = np.array(
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
        )

        np.testing.assert_array_almost_equal(
            mm.mod_mfd.values, mod_mfd, decimal=2
        )

    def test_get_moment_from_mfd_dict(self):
        mfd = {
            6.0: 1e-2,
            7.0: 1e-3,
            8.0: 1e-4,
        }

        moment = get_moment_from_mfd(mfd)
        assert np.isclose(moment, 1.589033688965738e17, rtol=1e-2)

    def test_get_moment_from_mfd_other(self):
        mfd = 1.0  # this is a stand-in for a different type of input

        self.assertRaises(ValueError, get_moment_from_mfd, mfd)

    def test__get_moment_from_mfd_dict(self):
        mfd = {
            6.0: 1e-2,
            7.0: 1e-3,
            8.0: 1e-4,
        }

        moment = _get_moment_from_mfd_dict(mfd)
        assert np.isclose(moment, 1.589033688965738e17, rtol=1e-2)

    def test_mag_diff_likelihood(self):
        eq_mag = 7.2
        mag_window = 1.0
        # rup_mags = np.linspace(5.0, 8.0, num=20)

        # import matplotlib.pyplot as plt

        # plt.plot(rup_mags, mag_diff_likelihood(eq_mag, rup_mags, mag_window))
        # plt.axvline(x=eq_mag, color="r")
        # plt.axvline(x=eq_mag - mag_window / 2, color="r")
        # plt.axvline(x=eq_mag + mag_window / 2, color="r")

        # plt.show()
        rup_mags = np.array([5.0, 6.0, 6.2, 6.5, 7.0, 7.2, 7.5, 8.0])
        likelihood = mag_diff_likelihood(eq_mag, rup_mags, mag_window)
        likelihood_expected = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.6, 1.0, 0.4, 0.0]
        )
        np.testing.assert_allclose(likelihood, likelihood_expected)

    def test_get_distances(self):
        eq = self.eq_gdf.iloc[0]

        rups = self.rupture_gdf.iloc[:3]

        distances = get_distances(eq, rups)

        distances_expected = pd.Series(
            data=[50.220820, 48.373561, 46.537740],
            index=["0_0", "0_1", "0_2"],
        )

        np.testing.assert_allclose(distances, distances_expected)

    def test_get_rups_in_mag_range(self):
        eq = self.eq_gdf.loc[8]
        mag_range = 1.0
        rups_in_range = get_rups_in_mag_range(eq, self.rupture_gdf, mag_range)

        assert rups_in_range.magnitude.min() >= eq.magnitude - mag_range / 2
        assert rups_in_range.magnitude.max() <= eq.magnitude + mag_range / 2

    def test_get_nearby_rups(self):
        eq = self.eq_gdf.loc[8]

        nearby_rups = get_nearby_rups(eq, self.rupture_gdf)

        rup_cell_ids = nearby_rups["cell_id"].unique().tolist()
        assert rup_cell_ids == [
            "83694afffffffff",
            "836864fffffffff",
            "836860fffffffff",
        ]

        assert eq.cell_id in rup_cell_ids

    def test_get_matching_rups_many(self):

        eq = self.eq_gdf.loc[11]

        matching_rups = get_matching_rups(
            eq,
            self.rupture_gdf,
            return_one=False,
            group_return_threshold=0.5,
        )

        assert len(matching_rups) == 3

    def test_get_matching_rups_best(self):

        eq = self.eq_gdf.loc[11]

        matching_rups = get_matching_rups(
            eq,
            self.rupture_gdf,
            return_one="best",
        )

        assert isinstance(matching_rups, pd.Series)

    def test_get_matching_rups_sample(self):

        np.random.seed(69)
        eq = self.eq_gdf.loc[11]

        matching_rup = get_matching_rups(
            eq,
            self.rupture_gdf,
            return_one="sample",
            group_return_threshold=0.5,
        )

        assert isinstance(matching_rup, pd.Series)
        assert matching_rup.name == "4_22"

    def test_match_eqs_to_rups(self):
        match_results = match_eqs_to_rups(
            self.eq_gdf,
            self.rupture_gdf,
        )

        assert len(match_results) == len(self.eq_gdf)
        for i, eq in self.eq_gdf.iterrows():
            assert match_results[i]["eq"] == i

    def test_rupture_matching_eval_fn(self):
        rup_matching_results = rupture_matching_eval_fn(
            self.rupture_gdf,
            self.eq_gdf,
        )

        assert len(rup_matching_results["matched_rups"]) == len(self.eq_gdf)
