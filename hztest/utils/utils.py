from multiprocessing import Pool
import os
from typing import Sequence, Dict, List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from .stats import sample_event_times_in_interval
from .bins import MagBin, SpacemagBin



def flatten_list(lol: List[list]) -> list:
    """
    Flattens a list of lists (lol).  

    >>> _flatten_list([['l'], ['o', 'l']])
    ['l', 'o', 'l']

    """
    return [item for sublist in lol for item in sublist]


def rupture_dict_from_logic_tree_dict(logic_tree_dict: dict, 
                                 source_types: Sequence[str]=('simple_fault'),
                                 parallel: bool=True,
                                 n_procs: Optional[int]=None
                                 ) -> dict:
    """
    Creates a dictionary of ruptures from a dictionary representation of a logic
    tree (as produced by :func:`~hztest.utils.io.process_source_logic_tree`).
    Each branch in the logic tree dict is a value (with the branch name as a
    key) and this structure is retained in the resulting rupture dict. However
    all of the ruptures from each source within a branch will be flattened to a
    single list.

    Use the `source_types` argument to specify which types of sources should be
    used to collect ruptures.

    :param logic_tree_dict:
        Seismic source logic tree

    :param source_types:
        Types of sources to collect ruptures from. Defaults to `simple_fault`
        but other values (`complex_fault`, `area`, `point`, `multipoint`) can
        also be given (or any combination of these).

    :param parallel:
        Flag to use a parallel input method (parallelizing with each source
        branch). Defaults to `True`.
    
    :param n_procs:
        Number of parallel processes. If `None` is passed, it defaults to
        `os.cpu_count() -1`. Only used if `parallel` is `True`.

    :returns:
        Ruptures from the logic tree collected into each logic tree branch.

    """

    if parallel is True:
        return {br: rupture_list_from_lt_branch_parallel(branch, source_types,
                                                         n_procs=n_procs)
                for br, branch in logic_tree_dict.items()}
    else:
        return {br: rupture_list_from_lt_branch(branch, source_types)
                for br, branch in logic_tree_dict.items()}
        

def rupture_list_from_lt_branch(branch: dict,
                                source_types: Sequence[str]=('simple_fault')
                                ) -> list:
    """
    Creates a list of ruptures from all of the sources within a single logic
    tree branch, adding the `source_id` of each source to the rupture as an
    attribute called `source`.

    :param branch:
        Logic tree branch from which to concatenate ruptures

    :param source_types:
        Types of sources to collect ruptures from. Defaults to `simple_fault`
        but other values (`complex_fault`, `area`, `point`, `multipoint`) can
        also be given (or any combination of these).

    :returns:
        All of the ruptures from all sources of `sources_types` in the logic
        tree branch.
    """

    rupture_list = []
    
    def process_rup(rup, source):
        rup.source = source.source_id
        return rup
        
    for source_type, sources in branch.items():
        if source_type in source_types and sources != []:
            rups = [process_rup(r, source) for source in sources
                                           for r in source.iter_ruptures()]

            rupture_list.extend(rups)

    return rupture_list

def _process_rup(rup, source):
    rup.source = source.source_id
    return rup

def _process_source(source):
    return [_process_rup(r, source) for r in source.iter_ruptures()]

def rupture_list_from_lt_branch_parallel(branch: dict,
                                source_types: Sequence[str]=('simple_fault'),
                                n_procs: Optional[int]=None
                                ) -> list:
    """
    Creates a list of ruptures from all of the sources within a single logic
    tree branch, adding the `source_id` of each source to the rupture as an
    attribute called `source`.

    Works in 

    :param branch:
        Logic tree branch from which to concatenate ruptures

    :param source_types:
        Types of sources to collect ruptures from. Defaults to `simple_fault`
        but other values (`complex_fault`, `area`, `point`, `multipoint`) can
        also be given (or any combination of these).

    :param n_procs:
        Number of parallel processes. If `None` is passed, it defaults to
        `os.cpu_count() -1`.

    :returns:
        All of the ruptures from all sources of `sources_types` in the logic
        tree branch.
    """

    if n_procs is None:
        n_procs = os.cpu_count() - 1

    rupture_list = []
    
    for source_type, sources in branch.items():
        if source_type in source_types and sources != []:
            with Pool(n_procs) as pool:
                rups = pool.map(_process_source, sources)
                rups = flatten_list(rups)
                rupture_list.extend(rups)

    return rupture_list


def rupture_list_to_gdf(rupture_list: list) -> gpd.GeoDataFrame:
    """
    Creates a 
    """
    
    df = pd.DataFrame(index=range(len(rupture_list)),
                      data=rupture_list, columns=['rupture'])

    df['geometry'] = df.apply(lambda z: Point(z.rupture.hypocenter.longitude, 
                                              z.rupture.hypocenter.latitude),
                              axis=1)
    return gpd.GeoDataFrame(df)


def make_spatial_bins_df_from_file(bin_fp):
    """
    Returns a geopandas dataframe from a file containing spatial bins as
    polygons.
    """

    bin_df = gpd.read_file(bin_fp)

    return bin_df


def add_ruptures_to_bins(rupture_gdf, bin_df, parallel=False):

    join_df = gpd.sjoin(rupture_gdf, bin_df, how='left')

    rupture_gdf['bin_id'] = join_df['index_right']

    def bin_row(row):
        if not np.isnan(row.bin_id): 
            spacemag_bin = bin_df.loc[row.bin_id, 'SpacemagBin']
            nearest_bc = _nearest_bin(row.rupture.mag, spacemag_bin.mag_bin_centers)
            spacemag_bin.mag_bins[nearest_bc].ruptures.append(row.rupture)
    if parallel is False:
        _ = rupture_gdf.apply(bin_row, axis=1)
        return

    def bin_row_apply(df):
        _ = df.apply(bin_row, axis=1)

    if parallel is True:
        raise NotImplementedError


def make_earthquake_gdf(earthquake_df):
    pass


def _nearest_bin(val, bin_centers):

    bca = np.array(bin_centers)

    return bin_centers[np.argmin(np.abs(val-bca))]


def add_earthquakes_to_bins(earthquake_gdf, bin_df):
    # observed_earthquakes, not ruptures
    # need to check for out-of-range events, i.e. those too big/small for bins

    join_df = gpd.sjoin(earthquake_gdf, bin_df, how='left')

    earthquake_gdf['bin_id'] = join_df['index_right']

    for i, eq in earthquake_gdf.iterrows():
        if not np.isnan(eq.bin_id): 
            spacemag_bin = bin_df.loc[eq.bin_id, 'SpacemagBin']
            nearest_bc = _nearest_bin(eq.Eq.mag, spacemag_bin.mag_bin_centers)

            spacemag_bin.mag_bins[nearest_bc].observed_earthquakes.append(
                                                                    eq['Eq'])
            spacemag_bin.observed_earthquakes[nearest_bc].append(eq['Eq'])


def make_SpacemagBins_from_bin_df(bin_df, min_mag=6., max_mag=9.,
                                  bin_width=0.1,):
    def bin_to_mag(row):
        return SpacemagBin(row.geometry, bin_id=row._name, min_mag=min_mag,
                            max_mag=max_mag)
    bin_df['SpacemagBin'] = bin_df.apply(bin_to_mag, axis=1)


class Earthquake():
    def __init__(self, mag=None, latitude=None, longitude=None, depth=None,
                 time=None, source=None, event_id=None):
        self.mag = mag
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.time = time
        self.source = source
        self.event_id = event_id


def make_earthquakes(rupture, interval_length, t0=0.):
    event_times = sample_event_times_in_interval(rupture.occurrence_rate,
                                                 interval_length, t0)
    try:
        source = rupture.source
    except:
        source = None
    
    eqs = [Earthquake(mag=rupture.mag, latitude=rupture.hypocenter.latitude,
                      longitude=rupture.hypocenter.longitude, 
                      depth=rupture.hypocenter.depth,
                      source=source, time=et)
                      for et in event_times]
    return eqs




