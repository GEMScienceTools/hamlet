import unittest
import os

import numpy as np
from shapely.geometry import Polygon
from openquake.hazardlib.source import ParametricProbabilisticRupture as Rupture
from openquake.hazardlib.geo import Point



import hztest
import hztest.model_test_frameworks.gem.gem_test_functions as gtu
from hztest.utils import SpacemagBin


#BASE_DATA_PATH = os.path.dirname(__file__)
#test_data_dir = os.path.join(BASE_DATA_PATH, '..', '..', 'data', '')

class TestMFDConstruction(unittest.TestCase):

    def setUp(self):

        self.spacemag_bin_1 = SpacemagBin(Polygon(), min_mag=6.0, max_mag=7.0,
                                        bin_width=0.1)

        self.spacemag_bin_1.mag_bins[6.0].ruptures.append(
            Rupture(6.0, 'undefined', None, Point(0.,0.), None, 0.01, None)
        )
        
        self.spacemag_bin_1.mag_bins[6.2].ruptures.append(
            Rupture(6.2, 'undefined', None, Point(0.,0.), None, 0.005, None)
        )

    def test_get_stochastic_mfd_counts(self):
        np.random.seed(69)
        stoch_mfd_counts = gtu.get_stochastic_mfd_counts(self.spacemag_bin_1, 
                                                         n_iters=10,
                                                         interval_length=100)

        stoch_mfd_counts_ = {6.0: [0, 1, 2, 1, 0, 1, 1, 2, 0, 2],
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
                             7.1: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
        
        self.assertEqual(stoch_mfd_counts, stoch_mfd_counts_)

    def test_get_stochcastic_mfds(self):
        np.random.seed(69)

        stoch_mfd = gtu.get_stochastic_mfd(self.spacemag_bin_1, 
                                           n_iters=10, interval_length=100)

        stoch_mfd_ = {6.0: {0: 0.3, 1: 0.4, 2: 0.3},
                      6.1: {0: 1.0},
                      6.2: {0: 0.7, 1: 0.3},
                      6.3: {0: 1.0},
                      6.4: {0: 1.0},
                      6.5: {0: 1.0},
                      6.6: {0: 1.0},
                      6.7: {0: 1.0},
                      6.8: {0: 1.0},
                      6.9: {0: 1.0},
                      7.0: {0: 1.0},
                      7.1: {0: 1.0}}

        self.assertEqual(stoch_mfd, stoch_mfd_)

if __name__ == '__main__':
    unittest.main()