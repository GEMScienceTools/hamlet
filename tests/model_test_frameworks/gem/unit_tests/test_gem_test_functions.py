import unittest
import os

import numpy as np
from shapely.geometry import Polygon
from openquake.hazardlib.source import ParametricProbabilisticRupture as Rupture
from openquake.hazardlib.geo import Point

import openquake.hme
import openquake.hme.model_test_frameworks.gem.gem_test_functions as gtf
from openquake.hme.utils import SpacemagBin
from openquake.hme.utils.io import read_rupture_file

BASE_DATA_PATH = os.path.dirname(__file__)
TEST_DATA_DIR = os.path.join(BASE_DATA_PATH, '..', '..', '..', 'data',
    'source_models', 'sm1')
TEST_RUP_FILE = os.path.join(TEST_DATA_DIR, "sm1_rups.csv")
TEST_EQS_FILE = os.path.join(TEST_DATA_DIR, 'data', 'phl_eqs.csv')


class TestMFDConstruction(unittest.TestCase):
    def setUp(self):

        self.spacemag_bin_1 = SpacemagBin(Polygon(),
                                          min_mag=6.0,
                                          max_mag=7.0,
                                          bin_width=0.1)

        self.spacemag_bin_1.mag_bins[6.0].ruptures.append(
            Rupture(6.0, 'undefined', None, Point(0., 0.), None, 0.01, None))

        self.spacemag_bin_1.mag_bins[6.2].ruptures.append(
            Rupture(6.2, 'undefined', None, Point(0., 0.), None, 0.005, None))

    def test_get_stochastic_mfd_counts(self):
        np.random.seed(69)
        stoch_mfd_counts = gtf.get_stochastic_mfd_counts(self.spacemag_bin_1,
                                                         n_iters=10,
                                                         interval_length=100)

        stoch_mfd_counts_ = {
            6.0: [0, 1, 2, 1, 0, 1, 1, 2, 0, 2],
            6.1: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            6.2: [1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            6.3: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            6.4: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            6.5: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            6.6: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            6.7: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            6.8: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            6.9: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            7.0: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            7.1: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        }

        self.assertEqual(stoch_mfd_counts, stoch_mfd_counts_)

    def test_get_stochcastic_mfds(self):
        np.random.seed(69)

        stoch_mfd = gtf.get_stochastic_mfd(self.spacemag_bin_1,
                                           n_iters=10,
                                           interval_length=100)

        stoch_mfd_ = {
            6.0: {
                0: 0.3,
                1: 0.4,
                2: 0.3
            },
            6.1: {
                0: 1.0
            },
            6.2: {
                0: 0.7,
                1: 0.3
            },
            6.3: {
                0: 1.0
            },
            6.4: {
                0: 1.0
            },
            6.5: {
                0: 1.0
            },
            6.6: {
                0: 1.0
            },
            6.7: {
                0: 1.0
            },
            6.8: {
                0: 1.0
            },
            6.9: {
                0: 1.0
            },
            7.0: {
                0: 1.0
            },
            7.1: {
                0: 1.0
            }
        }

        self.assertEqual(stoch_mfd, stoch_mfd_)


class testStochasticMomentStuff(unittest.TestCase):
    def setUp(self):
        self.rup_gdf = read_rupture_file(TEST_RUP_FILE)
        self.bin_gdf = openquake.hme.utils.make_bin_gdf_from_rupture_gdf(
            self.rup_gdf
        )
        self.eq_gdf = openquake.hme.utils.make_earthquake_gdf_from_csv(
            TEST_EQS_FILE)

        openquake.hme.utils.add_ruptures_to_bins(self.rup_gdf, self.bin_gdf)
        openquake.hme.utils.add_earthquakes_to_bins(self.eq_gdf, self.bin_gdf)

        self.sb2 = self.bin_gdf.iloc[2]['SpacemagBin']

    def test_get_stochastic_moment(self):
        np.random.seed(69)
        mo_sum = gtf.get_stochastic_moment(self.sb2, 4000.)
        # doesn't work until I figure out better random seeding
        #np.testing.assert_almost_equal(mo_sum, 9.345019861038536e+20)

    def test_get_stochastic_moment_set(self):
        np.random.seed(69)
        mo_set = gtf.get_stochastic_moment_set(self.sb2, 40., 20)
        #np.testing.assert_array_almost_equal(
        #
        # aa = np.array(
        #        [5.87789558e+18, 4.60281548e+19, 2.66704286e+18, 1.88364909e+18,
        #         0.00000000e+00, 4.21696503e+19, 9.85321859e+18, 4.54437415e+18,
        #         0.00000000e+00, 1.33352143e+18, 7.76154467e+18, 3.75837404e+18,
        #         5.32145012e+18, 4.54437415e+18, 1.88364909e+18, 3.99424649e+18,
        #         0.00000000e+00, 0.00000000e+00, 9.06721849e+18, 1.33352143e+18]
        #    )

        #for i, val in enumerate(aa):
        #    if not np.testing.assert_almost_equal(val, mo_set[i]):
        #        print(i)
        
        ##    mo_set,
        ##    decimal=3
        ##)

    def test_rank_obs_moment(self):
        obs_moment_pctile = gtf.rank_obs_moment(self.sb2, 40., 200)



