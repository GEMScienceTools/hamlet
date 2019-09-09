import os

import unittest
#import sys; sys.path.append('../')

from openquake.hazardlib.source import SimpleFaultSource
from openquake.hazardlib.source.rupture import ParametricProbabilisticRupture

import hztest

BASE_PATH = os.path.dirname(__file__)
test_data_dir = os.path.join(BASE_PATH, 'data', 'source_models', 'sm1')


class TestBasicUtils(unittest.TestCase):
    def test_flatten_list(self):
        lol = [['l'], ['o'], ['l']]
        flol = hztest.utils.flatten_list(lol)
        self.assertEqual(flol, ['l', 'o', 'l'])


class TestPHL1(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print('setting up')

        print(BASE_PATH)

        self.test_dir = test_data_dir + "/"

        self.lt = hztest.utils.io.process_source_logic_tree(self.test_dir)

        self.rup_dict = hztest.utils.rupture_dict_from_logic_tree_dict(self.lt)

        self.rup_list = hztest.utils.rupture_list_from_lt_branch(self.lt['b1'])

        self.rup_gdf = hztest.utils.rupture_list_to_gdf(self.rup_list)
        self.bin_df = hztest.utils.make_SpacemagBins_from_bin_gis_file(
            self.test_dir + 'data/phl_f_bins.geojson')
        #self.eq_df = hztest.utils.make_earthquake_gdf_from_csv(
        #    self.test_dir + 'data/phl_eqs.csv')

    def test_process_source_logic_tree(self):
        test_lt = {
            'b1': {
                'area': [],
                'complex_fault': [],
                'point': [],
                'multipoint': [],
                'none': [None]
            }
        }

        for k, v in test_lt['b1'].items():
            self.assertEqual(self.lt['b1'][k], v)

        self.assertIsInstance(self.lt['b1']['simple_fault'][0],
                              SimpleFaultSource)

    def test_rupture_dict_from_logic_dict(self):
        self.assertEqual(list(self.rup_dict.keys()), ['b1'])
        self.assertEqual(len(self.rup_dict['b1']), 7797)
        self.assertIsInstance(self.rup_dict['b1'], list)
        self.assertIsInstance(
            self.rup_dict['b1'][0],
            #ParametricProbabilisticRupture)
            hztest.utils.SimpleRupture)

    def test_rupture_list_from_lt_branch(self):
        self.assertIsInstance(self.rup_list, list)
        self.assertEqual(len(self.rup_list), 7797)
        self.assertIsInstance(
            self.rup_list[0],
            #ParametricProbabilisticRupture)
            hztest.utils.SimpleRupture)

    def test_rupture_list_from_lt_branch_parallel(self):

        self.rup_list_par = hztest.utils.rupture_list_from_lt_branch_parallel(
            self.lt['b1'], n_procs=4)

        self.assertIsInstance(self.rup_list_par, list)
        self.assertEqual(len(self.rup_list_par), 7797)
        self.assertIsInstance(
            self.rup_list_par[0],
            #ParametricProbabilisticRupture)
            hztest.utils.SimpleRupture)

    @unittest.skip('not implemented')
    def test_rupture_list_to_gdf(self):
        pass

    def test_make_spatial_bins_from_file(self):
        self.assertEqual(self.bin_df.loc[0].geometry.area, 0.07185377067626442)

    def test_add_ruptures_to_bins(self):
        self.bin_df = hztest.utils.make_SpacemagBins_from_bin_gis_file(
            self.test_dir + 'data/phl_f_bins.geojson')
        self.eq_df = hztest.utils.make_earthquake_gdf_from_csv(
            self.test_dir + 'data/phl_eqs.csv')

        hztest.utils.add_ruptures_to_bins(self.rup_gdf,
                                          self.bin_df,
                                          parallel=False)
        num_rups_first_bin = len(
            self.bin_df.SpacemagBin.loc[0].mag_bins[6.2].ruptures)

        self.assertEqual(num_rups_first_bin, 209)

    def test_add_ruptures_to_bins_parallel(self):
        self.bin_df = hztest.utils.make_SpacemagBins_from_bin_gis_file(
            self.test_dir + 'data/phl_f_bins.geojson')
        self.eq_df = hztest.utils.make_earthquake_gdf_from_csv(
            self.test_dir + 'data/phl_eqs.csv')

        hztest.utils.add_ruptures_to_bins(self.rup_gdf,
                                          self.bin_df,
                                          parallel=True)
        num_rups_first_bin = len(
            self.bin_df.SpacemagBin.loc[0].mag_bins[6.2].ruptures)

        self.assertEqual(num_rups_first_bin, 209)

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
