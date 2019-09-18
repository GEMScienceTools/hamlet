import sys
sys.path.append('../')

import openquake.hme

import pandas as pd
import geopandas as gpd
import datetime
from shapely.geometry import Point
import numpy as np

phl_ssm_dir = '/Users/itchy/research/gem/hazard_models/mosaic/PHL/in/'
#phl_eq_data = './data/phl_test/phl_eq_cat.csv'
phl_eq_data = './data/phl_test/cat_dec_af_crustal_gr.csv'

bin_gj = './data/phl_test/phl_bins.geojson'

# bin stuff
# make bins
print('making bins')
bin_df = openquake.hme.utils.make_spatial_bins_df_from_file(bin_gj)
openquake.hme.utils.make_SpacemagBins_from_bin_df(bin_df)

# read earthquake catalog and put in bins
print('doing eq cat')
eq_df = pd.read_csv(phl_eq_data)
eq_df['time'] = eq_df.apply(lambda x: datetime.datetime(*np.int_((
    x.year, x.month, x.day, x.hour, x.minute, x.second))),
                            axis=1)
eq_df['geometry'] = eq_df.apply(lambda x: Point(x.longitude, x.latitude),
                                axis=1)
eq_df = gpd.GeoDataFrame(eq_df)
eq_df['Eq'] = eq_df.apply(lambda x: openquake.hme.utils.Earthquake(
    mag=x.magnitude,
    latitude=x.latitude,
    longitude=x.longitude,
    depth=x.depth,
    time=x.time,
    source=x.Agency,
    event_id=x.eventID),
                          axis=1)

openquake.hme.utils.add_earthquakes_to_bins(eq_df, bin_df)

print('processing sources')
print('    reading and sorting logic tree')
phlt = openquake.hme.utils.io.process_logic_tree(phl_ssm_dir)

print('    getting ruptures from sources')
rl = openquake.hme.utils.rupture_list_from_lt_branch(phlt['b1'],
                                              source_types=('simple_fault',
                                                            'point'))

print('    binning ruptures')
rgdf = openquake.hme.utils.rupture_list_to_gdf(rl)
openquake.hme.utils.add_ruptures_to_bins(rgdf, bin_df)
print('done processing sources.')

print('')

big_bins = [
    row['SpacemagBin'] for i, row in bin_df.iterrows()
    if sum(len(mb.ruptures) for mb in row['SpacemagBin'].mag_bins.values()) > 0
]

big_bins[0].sample_ruptures(50000)
