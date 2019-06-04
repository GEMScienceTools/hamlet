import unittest
#import sys; sys.path.append('../')

from openquake.hazardlib.source import SimpleFaultSource
from openquake.hazardlib.source.rupture import ParametricProbabilisticRupture

import hztest


class TestBasicUtils(unittest.TestCase):

    def test_flatten_list(self):
        lol = [['l'], ['o'], ['l']]
        flol = hztest.utils.flatten_list(lol)
        self.assertEqual(flol, ['l', 'o', 'l'])

class TestPHL1(unittest.TestCase):

    def setUp(self):
        self.test_dir = './data/source_models/sm1/'
        self.lt = hztest.utils.io.process_source_logic_tree(self.test_dir)

    def test_process_source_logic_tree(self):
        test_lt = {'b1': {'area': [],
                          'complex_fault': [],
                          'point': [],
                          'multipoint': [],
                          'none': [None]
                          }}

        for k, v in test_lt['b1'].items():
            self.assertEqual(self.lt['b1'][k], v)

        self.assertIsInstance(self.lt['b1']['simple_fault'][0], 
                              SimpleFaultSource)

    def test_rupture_dict_from_logic_dict(self):
        
        rup_dict = hztest.utils.rupture_dict_from_logic_tree_dict(self.lt)

        self.assertEqual(list(rup_dict.keys()), ['b1'])
        self.assertEqual(len(rup_dict['b1']), 7797)
        self.assertIsInstance(rup_dict['b1'], list)
        self.assertIsInstance(rup_dict['b1'][0], ParametricProbabilisticRupture)

    def test_rupture_list_from_lt_branch(self):
        rup_list = hztest.utils.rupture_list_from_lt_branch(self.lt['b1'])
        self.assertIsInstance(rup_list, list)
        self.assertEqual(len(rup_list), 7797)
        self.assertIsInstance(rup_list[0], ParametricProbabilisticRupture)

    def rupture_list_from_lt_branch_parallel(self):
        rup_list = hztest.utils.rupture_list_from_lt_branch_parallel(
            self.lt['b1'], n_procs=4)
        self.assertIsInstance(rup_list, list)
        self.assertEqual(len(rup_list), 7797)
        self.assertIsInstance(rup_list[0], ParametricProbabilisticRupture)

    @unittest.skip('not implemented')
    def test_rupture_list_to_gdf(self):
        pass

    @unittest.skip('not implemented')
    def test_make_spatial_bins_from_file(self):
        pass

    @unittest.skip('not implemented')
    def test_add_ruptures_to_bins(self):
        pass

    @unittest.skip('not implemented')
    def test_make_earthquake_gdf(self):
        pass

    @unittest.skip('not implemented')
    def test_nearest_bin(self):
        pass

    @unittest.skip('not implemented')
    def test_add_earthquakes_to_bins(self):
        pass

    @unittest.skip('not implemented')
    def test_make_SpacemagBins_from_bin_df(self):
        pass





if __name__ == '__main__':
    unittest.main()