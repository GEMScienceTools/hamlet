import os
import unittest

from openquake.hazardlib.source import SimpleFaultSource
from openquake.hazardlib.source.rupture import ParametricProbabilisticRupture

from openquake import hme

BASE_PATH = os.path.dirname(__file__)
test_data_dir = os.path.join(BASE_PATH, 'data', 'source_models', 'sm1')


class TestBasicUtils(unittest.TestCase):
    def test_flatten_list(self):
        lol = [['l'], ['o'], ['l']]
        flol = hme.utils.flatten_list(lol)
        self.assertEqual(flol, ['l', 'o', 'l'])


class TestPHL1(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        print('setting up')

        print(BASE_PATH)

        self.test_dir = test_data_dir + "/"

        self.lt = hme.utils.io.process_source_logic_tree(self.test_dir)

        self.rup_dict = hme.utils.rupture_dict_from_logic_tree_dict(self.lt)

        self.rup_list = hme.utils.rupture_list_from_source_list(self.lt['b1'])

        self.rup_gdf = hme.utils.rupture_list_to_gdf(self.rup_list)
        self.bin_df = hme.utils.make_bin_gdf_from_rupture_gdf(self.rup_gdf,
                                                              res=3,
                                                              parallel=False)

    def test_process_source_logic_tree(self):

        self.assertIsInstance(self.lt['b1'][0], SimpleFaultSource)

    def test_rupture_dict_from_logic_dict(self):
        self.assertEqual(list(self.rup_dict.keys()), ['b1'])
        self.assertEqual(len(self.rup_dict['b1']), 7797)
        self.assertIsInstance(self.rup_dict['b1'], list)
        self.assertIsInstance(
            self.rup_dict['b1'][0],
            #ParametricProbabilisticRupture)
            hme.utils.SimpleRupture)

    def test_rupture_list_from_lt_branch(self):
        self.assertIsInstance(self.rup_list, list)
        self.assertEqual(len(self.rup_list), 7797)
        self.assertIsInstance(
            self.rup_list[0],
            #ParametricProbabilisticRupture)
            hme.utils.SimpleRupture)

    def test_rupture_list_from_lt_branch_parallel(self):

        self.rup_list_par = hme.utils.rupture_list_from_source_list_parallel(
            self.lt['b1'], n_procs=4)

        self.assertIsInstance(self.rup_list_par, list)
        self.assertEqual(len(self.rup_list_par), 7797)
        self.assertIsInstance(
            self.rup_list_par[0],
            #ParametricProbabilisticRupture)
            hme.utils.SimpleRupture)

    def test_rupture_list_to_gdf(self):
        r0 = self.rup_gdf.loc[0]
        self.assertEqual(r0.bin_id, "83694afffffffff")
        self.assertEqual(r0.mag_r, 6)

    def test_make_bin_gdf_from_rupture_gdf(self):
        bin_df = hme.utils.make_bin_gdf_from_rupture_gdf(self.rup_gdf,
                                                         res=3,
                                                         parallel=False)
        self.assertEqual(bin_df.loc['836860fffffffff'].geometry.area,
                         1.0966917348958416)

    def test_make_spatial_bins_from_file(self):
        bin_df = hme.utils.make_SpacemagBins_from_bin_gis_file(
            self.test_dir + 'data/phl_f_bins.geojson')
        self.assertEqual(bin_df.loc[0].geometry.area, 0.07185377067626442)

    def test_add_ruptures_to_bins(self):
        hme.utils.add_ruptures_to_bins(self.rup_gdf, self.bin_df)

        num_rups_first_bin = len(self.bin_df.SpacemagBin.
                                 loc['836860fffffffff'].mag_bins[6.2].ruptures)

        self.assertEqual(num_rups_first_bin, 561)

    def test_make_earthquake_gdf(self):
        self.eq_df = hme.utils.make_earthquake_gdf_from_csv(self.test_dir +
                                                            'data/phl_eqs.csv')
        self.assertEqual(self.eq_df.loc[0].magnitude, 7.4)

    def test_add_earthquakes_to_bins(self):
        self.eq_df = hme.utils.make_earthquake_gdf_from_csv(self.test_dir +
                                                            'data/phl_eqs.csv')
        hme.utils.add_earthquakes_to_bins(self.eq_df, self.bin_df)
        sbin = self.bin_df.loc['836860fffffffff'].SpacemagBin
        self.assertEqual(len(sbin.mag_bins[6.0].observed_earthquakes), 2)

    def test_make_SpacemagBins_from_bin_gdf(self):
        self.assertIsInstance(self.bin_df.iloc[0].SpacemagBin,
                              hme.utils.SpacemagBin)


if __name__ == '__main__':
    unittest.main()
