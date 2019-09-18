import sys
sys.path.append('../')

import openquake.hme

import pandas as pd
import geopandas as gpd
import datetime
import time
from shapely.geometry import Point
import numpy as np
#import tqdm

#import dask.dataframe as dd
#import dask.multiprocessing
#dask.config.set(scheduler='processes')
#from dask.distributed import Client
#client = Client()

from multiprocessing import cpu_count, Pool

#phl_ssm_dir = '../../../hazard_models/PHL/in/'
phl_ssm_dir = '../../../hazard_models/mosaic/PHL/in/'
#phl_eq_data = './data/phl_test/phl_eq_cat.csv'
phl_eq_data = './data/phl_test/ISC-Phivolcs_dec_crustal_uh.csv'

bin_gj = './data/phl_test/phl_bins.geojson'

# bin stuff
# make bins
print('making bins')

min_bin_mag = 6.0
bin_width = 0.1

bin_df = openquake.hme.utils.make_spatial_bins_df_from_file(bin_gj)
openquake.hme.utils.make_SpacemagBins_from_bin_df(bin_df,
                                           min_mag=min_bin_mag,
                                           bin_width=bin_width)

# read earthquake catalog and put in bins
print('doing eq cat')
eq_df = pd.read_csv(phl_eq_data)

eq_df = eq_df[eq_df.magnitude > min_bin_mag]

eq_df['time'] = eq_df.apply(lambda x: datetime.datetime(*np.int_((
    x.year, x.month, x.day, x.hour, x.minute, x.second))),
                            axis=1)
eq_df['geometry'] = eq_df.apply(lambda x: Point(x.longitude, x.latitude),
                                axis=1)
eq_df = gpd.GeoDataFrame(eq_df)
eq_df.crs = {'init': 'epsg:4326'}
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
phlt = openquake.hme.utils.io.process_source_logic_tree(phl_ssm_dir)

print('    getting ruptures from sources')
t0 = time.time()
rl = openquake.hme.utils.rupture_list_from_lt_branch_parallel(
    phlt['b1'], source_types=('simple_fault', 'point'))
t1 = time.time()
print('     done in {:.1f} s'.format(t1 - t0))

t2 = time.time()
print('    binning ruptures')

rl = [r for r in rl if r.mag >= min_bin_mag - bin_width / 2.]

rgdf = openquake.hme.utils.rupture_list_to_gdf(rl)
rgdf.crs = {'init': 'epsg:4326'}
openquake.hme.utils.add_ruptures_to_bins(rgdf, bin_df, parallel=False)
t3 = time.time()
print('     done in {:.1f} s'.format(t3 - t2))
print('done processing sources.')

print('culling bins for ones with ruptures')

source_list = []
for i, row in bin_df.iterrows():
    cum_mfd = row.SpacemagBin.get_rupture_mfd(cumulative=True)
    if sum(cum_mfd.values()) > 0:
        source_list.append(i)

source_bin_df = bin_df.loc[source_list]
t4 = time.time()
print('    done in {:.1f} s'.format(t4 - t3))

print('calculating stochastic MFDs')

mfd_iters = 1000

bcs = bin_df.iloc[0].SpacemagBin.mag_bin_centers


def get_mfd_freq_counts(bc, eq_counts):
    n_eqs, n_occurrences = np.unique(eq_counts, return_counts=True)
    mfd_freq_counts = {
        n_eq: n_occurrences[i] / sum(n_occurrences)
        for i, n_eq in enumerate(n_eqs)
    }
    return mfd_freq_counts


def update_mfd_counts(mfd_counts, sb, interval_length):
    [
        mfd_counts[bc].append(int(n_eqs))
        for bc, n_eqs in sb.get_rupture_sample_mfd(
            interval_length, normalize=False, cumulative=False).items()
    ]


def get_stoch_mfd(sb, interval_length=40, iters=mfd_iters, bcs=bcs):
    mfd_counts = {bc: [] for bc in bcs}

    _ = [
        update_mfd_counts(mfd_counts, sb, interval_length=interval_length)
        for ii in range(iters)
    ]

    mfd_freq_counts = {}

    for bc, eq_counts in mfd_counts.items():
        n_eqs, n_occurrences = np.unique(eq_counts, return_counts=True)
        mfd_freq_counts[bc] = {
            n_eq: n_occurrences[i] / sum(n_occurrences)
            for i, n_eq in enumerate(n_eqs)
        }
    return mfd_freq_counts


#source_bin_mfds = source_bin_df['SpacemagBin'].apply(get_stoch_mfd)

#sbins = dd.from_pandas(source_bin_df['SpacemagBin'], npartitions=30)
sbin_meta = pd.Series(name='SpacemagBin', dtype=object)
#source_bin_mfds = sbins.apply(get_stoch_mfd, meta=sbin_meta)

#cores = cpu_count()-1 #Number of CPU cores on your system
cores = 30
partitions = cores  #Define as many partitions as you want


def parallelize(data, func):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    result = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return result


def stoch_mfd_apply(series):
    return series.apply(get_stoch_mfd)


source_bin_mfds = parallelize(source_bin_df['SpacemagBin'], stoch_mfd_apply)

#source_bin_mfds = dd.from_pandas(source_bin_df['SpacemagBin'],
#                                 npartitions=90).map_partitions(
#                                     lambda s: s.apply(get_stoch_mfd),
#                                                       meta=sbin_meta
#                                 ).compute()

t5 = time.time()
print('    done in {:.1f} s'.format(t5 - t4))

print('calculating log likelihoods')


def calc_bin_likelihood(bc, n_eqs, mfd_dict):
    try:
        return mfd_dict[bc][n_eqs]
    except KeyError:
        return 1e-5


def calc_mfd_log_likelihood(obs_eqs, mfd_dict):
    n_bins = len(obs_eqs.keys())
    return np.exp(
        np.sum(
            np.log([
                calc_bin_likelihood(bc, len(eqs), mfd_dict)
                for bc, eqs in obs_eqs.items()
            ])) / n_bins)


def calc_row_log_likelihood(row, mfd_df=source_bin_mfds):

    obs_eqs = row.SpacemagBin.observed_earthquakes
    mfd_dict = mfd_df.loc[row._name]

    return calc_mfd_log_likelihood(obs_eqs, mfd_dict)


source_bin_df['log_like'] = source_bin_df.apply(calc_row_log_likelihood,
                                                axis=1)

t6 = time.time()
print('    done in {:.1f} s'.format(t6 - t5))

print('done with everything in {:.1f} s'.format(t6 - t0))

source_bin_df['idx'] = source_bin_df.index.values

source_bin_df[['idx', 'log_like', 'geometry'
               ]].to_file('/home/itchy/Desktop/phl_bin_likes.geojson',
                          driver='GeoJSON')
