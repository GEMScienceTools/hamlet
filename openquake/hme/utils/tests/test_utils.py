import os
import unittest

import numpy as np
from openquake.hazardlib.source import SimpleFaultSource
from openquake.hazardlib.source.rupture import ParametricProbabilisticRupture

from openquake.hme.utils.io import process_source_logic_tree
from openquake.hme.utils import (
    flatten_list,
    rupture_dict_from_logic_tree_dict,
    rupture_list_from_source_list,
    rupture_list_from_source_list_parallel,
    rupture_list_to_gdf,
    SimpleRupture,
    SpacemagBin,
    make_bin_gdf_from_rupture_gdf,
    make_SpacemagBins_from_bin_gis_file,
    make_SpacemagBins_from_bin_gdf,
    add_ruptures_to_bins,
    add_earthquakes_to_bins,
    make_earthquake_gdf_from_csv,
    get_model_mfd,
    get_obs_mfd,
    get_total_obs_eqs,
    get_model_annual_eq_rate,
)

BASE_PATH = os.path.dirname(__file__)
test_data_dir = os.path.join(BASE_PATH, "data", "source_models", "sm1")


class TestBasicUtils(unittest.TestCase):
    def test_flatten_list(self):
        lol = [["l"], ["o"], ["l"]]
        flol = flatten_list(lol)
        self.assertEqual(flol, ["l", "o", "l"])


class TestPHL1(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.test_dir = test_data_dir + "/"

        self.lt, self.weights = process_source_logic_tree(self.test_dir)

        self.rup_dict = rupture_dict_from_logic_tree_dict(self.lt)

        self.rup_list = rupture_list_from_source_list(self.lt["b1"])

        self.rup_gdf = rupture_list_to_gdf(self.rup_list)
        self.bin_gdf = make_bin_gdf_from_rupture_gdf(self.rup_gdf,
                                                     h3_res=3,
                                                     parallel=False)

    def test_process_source_logic_tree(self):

        self.assertIsInstance(self.lt["b1"][0], SimpleFaultSource)

    def test_rupture_dict_from_logic_dict(self):
        self.assertEqual(list(self.rup_dict.keys()), ["b1"])
        self.assertEqual(len(self.rup_dict["b1"]), 7797)
        self.assertIsInstance(self.rup_dict["b1"], list)
        self.assertIsInstance(
            self.rup_dict["b1"][0],
            # ParametricProbabilisticRupture)
            SimpleRupture,
        )

    def test_rupture_list_from_lt_branch(self):
        self.assertIsInstance(self.rup_list, list)
        self.assertEqual(len(self.rup_list), 7797)
        self.assertIsInstance(
            self.rup_list[0],
            # ParametricProbabilisticRupture)
            SimpleRupture,
        )

    def test_rupture_list_from_lt_branch_parallel(self):

        self.rup_list_par = rupture_list_from_source_list_parallel(
            self.lt["b1"], n_procs=4)

        self.assertIsInstance(self.rup_list_par, list)
        self.assertEqual(len(self.rup_list_par), 7797)
        self.assertIsInstance(
            self.rup_list_par[0],
            # ParametricProbabilisticRupture)
            SimpleRupture,
        )

    def test_rupture_list_to_gdf(self):
        r0 = self.rup_gdf.loc[0]
        self.assertEqual(r0.bin_id, "83694afffffffff")
        self.assertEqual(r0.mag_r, 6)

    def test_make_bin_gdf_from_rupture_gdf(self):
        bin_df = make_bin_gdf_from_rupture_gdf(self.rup_gdf,
                                               h3_res=3,
                                               parallel=False)
        np.testing.assert_almost_equal(
            bin_df.loc["836860fffffffff"].geometry.area, 1.0966917348958416)

    def test_make_spatial_bins_from_file(self):
        bin_df = make_SpacemagBins_from_bin_gis_file(self.test_dir +
                                                     "data/phl_f_bins.geojson")
        self.assertEqual(bin_df.loc[0].geometry.area, 0.07185377067626442)

    def test_add_ruptures_to_bins(self):
        add_ruptures_to_bins(self.rup_gdf, self.bin_gdf)

        num_rups_first_bin = len(self.bin_gdf.SpacemagBin.
                                 loc["836860fffffffff"].mag_bins[6.2].ruptures)

        self.assertEqual(num_rups_first_bin, 561)

    def test_make_earthquake_gdf(self):
        self.eq_df = make_earthquake_gdf_from_csv(self.test_dir +
                                                  "data/phl_eqs.csv")
        self.assertEqual(self.eq_df.loc[0].magnitude, 7.4)

    def test_add_earthquakes_to_bins(self):
        self.eq_df = make_earthquake_gdf_from_csv(self.test_dir +
                                                  "data/phl_eqs.csv")
        add_earthquakes_to_bins(self.eq_df, self.bin_gdf)
        sbin = self.bin_gdf.loc["836860fffffffff"].SpacemagBin
        self.assertEqual(len(sbin.mag_bins[6.0].observed_earthquakes), 2)

    def test_make_SpacemagBins_from_bin_gdf(self):
        self.assertIsInstance(self.bin_gdf.iloc[0].SpacemagBin, SpacemagBin)

    def test_get_model_mfd_noncum(self):
        mod_mfd = get_model_mfd(self.bin_gdf)
        mod_mfd_res = {
            6.0: 0.0814075600000007,
            6.2: 0.1160290199999997,
            6.4: 0.0732093600000006,
            6.6: 0.04619198000000058,
            6.8: 0.02914517000000005,
            7.0: 0.01838935999999996,
            7.2: 0.011602900000000006,
            7.4: 0.007320929999999995,
            7.6: 0.004619199999999999,
            7.8: 0.00162429,
            8.0: 0,
            8.2: 0,
            8.4: 0,
            8.6: 0,
            8.8: 0,
            9.0: 0,
            9.2: 0,
        }

        for k, v in mod_mfd.items():
            np.testing.assert_almost_equal(v, mod_mfd_res[k])

    def test_get_model_mfd_cum(self):
        mod_mfd = get_model_mfd(self.bin_gdf, cumulative=True)
        mod_mfd_res = {
            6.0: 0.38953977,
            6.2: 0.30813221,
            6.4: 0.19210319,
            6.6: 0.11889383,
            6.8: 0.07270185,
            7.0: 0.04355668,
            7.2: 0.02516732,
            7.4: 0.01356442,
            7.6: 0.00624349,
            7.8: 0.00162429,
            8.0: 0,
            8.2: 0,
            8.4: 0,
            8.6: 0,
            8.8: 0,
            9.0: 0,
            9.2: 0,
        }

        for k, v in mod_mfd.items():
            np.testing.assert_almost_equal(v, mod_mfd_res[k])

    def test_get_obs_mfd_noncum(self):
        obs_mfd = get_obs_mfd(self.bin_gdf, t_yrs=40.0, cumulative=False)
        obs_mfd_res = {
            6.0: 0.05,
            6.2: 0.025,
            6.4: 0.025,
            6.6: 0.0,
            6.8: 0.0,
            7.0: 0.0,
            7.2: 0.0,
            7.4: 0.025,
            7.6: 0.025,
            7.8: 0.0,
            8.0: 0.0,
            8.2: 0.0,
            8.4: 0.0,
            8.6: 0.0,
            8.8: 0.0,
            9.0: 0.0,
            9.2: 0.0,
        }

        for k, v in obs_mfd.items():
            np.testing.assert_almost_equal(v, obs_mfd_res[k])

    def test_get_obs_mfd_cum(self):
        obs_mfd = get_obs_mfd(self.bin_gdf, t_yrs=40.0, cumulative=True)
        obs_mfd_res = {
            6.0: 0.15,
            6.2: 0.1,
            6.4: 0.075,
            6.6: 0.05,
            6.8: 0.05,
            7.0: 0.05,
            7.2: 0.05,
            7.4: 0.05,
            7.6: 0.025,
            7.8: 0.0,
            8.0: 0.0,
            8.2: 0.0,
            8.4: 0.0,
            8.6: 0.0,
            8.8: 0.0,
            9.0: 0.0,
            9.2: 0.0,
        }

        for k, v in obs_mfd.items():
            np.testing.assert_almost_equal(v, obs_mfd_res[k])

    def test_get_model_annual_eq_rate(self):
        ann_eq_rate = get_model_annual_eq_rate(self.bin_gdf)
        rate = 0.389539775
        np.testing.assert_almost_equal(ann_eq_rate, rate)

    def test_get_total_obs_eqs(self):
        obs_eqs = get_total_obs_eqs(self.bin_gdf)
        # just test some aspects instead of instantiating Earthquakes
        for eq in obs_eqs:
            assert eq.magnitude in [7.4, 7.54, 6.07, 5.95, 6.26, 6.44]
        assert len(obs_eqs) == 6
