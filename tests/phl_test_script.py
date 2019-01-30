import sys; sys.path.append('../')

import hztest

import pandas as pd
import geopandas as gpd
import datetime
from shapely.geometry import Point
import numpy as np

phl_ssm_dir = '/Users/itchy/research/gem/hazard_models/mosaic/PHL/in/'
phl_eq_data = './data/phl_test/phl_eq_cat.csv'

bin_gj = './data/phl_test/phl_bins.geojson'


# bin stuff
# make bins
print('making bins')
bin_df = hztest.utils.make_spatial_bins_df_from_file(bin_gj)
hztest.utils.make_SpacemagBins_from_bin_df(bin_df)

# read earthquake catalog and put in bins
print('doing eq cat')
eq_df = pd.read_csv(phl_eq_data)
eq_df['time'] = eq_df.apply(lambda x: datetime.datetime(
                                        *np.int_((x.Year, x.Month, x.Day,
                                                  x.Hour, x.Minute, x.Second))),
                            axis=1)
eq_df['geometry'] = eq_df.apply(lambda x: Point(x.Longitude, x.Latitude), axis=1)
eq_df = gpd.GeoDataFrame(eq_df)
eq_df['Eq'] = eq_df.apply(lambda x: hztest.utils.Earthquake(
                              mag=x.Mw, latitude=x.Latitude,
                              longitude=x.Longitude, depth=x.Depth, time=x.time,
                              source='phl_catalog', event_id=x.Event_ID),
                         axis=1)

hztest.utils.add_earthquakes_to_bins(eq_df, bin_df)


print('processing sources')
print('    reading and sorting logic tree')
phlt = hztest.utils.io.process_logic_tree(phl_ssm_dir)

print('    getting ruptures from sources')
rl = hztest.utils.rupture_list_from_lt_branch(phlt['b1'])

print('    binning ruptures')
rgdf = hztest.utils.rupture_list_to_gdf(rl)
hztest.utils.add_ruptures_to_bins(rgdf, bin_df)
print('done processing sources.')

print('')


big_bins = [row['SpacemagBin'] for i, row in bin_df.iterrows()
            if sum(len(mb.ruptures) 
            for mb in row['SpacemagBin'].mag_bins.values()) > 0]


big_bins[0].sample_ruptures(50000)