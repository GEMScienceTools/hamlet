import os
import datetime
from multiprocessing import Pool
from typing import Sequence, Dict, List, Optional, Union, Tuple

import attr
import dateutil
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from openquake.hazardlib.source.rupture import (
    NonParametricProbabilisticRupture, ParametricProbabilisticRupture)

from .stats import sample_event_times_in_interval
from .bins import MagBin, SpacemagBin

_n_procs = max(1, os.cpu_count() - 1)


def _parallelize(data, func, cores: int = _n_procs,
                 partitions: int = _n_procs):
    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    result = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return result


def flatten_list(lol: List[list]) -> list:
    """
    Flattens a list of lists (lol).  

    >>> flatten_list([['l'], ['o', 'l']])
    ['l', 'o', 'l']

    """
    return [item for sublist in lol for item in sublist]


def rupture_dict_from_logic_tree_dict(
        logic_tree_dict: dict,
        source_types: Sequence[str] = ('simple_fault'),
        parallel: bool = True,
        n_procs: Optional[int] = None) -> dict:
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
        return {
            br: rupture_list_from_lt_branch_parallel(branch,
                                                     source_types,
                                                     n_procs=n_procs)
            for br, branch in logic_tree_dict.items()
        }
    else:
        return {
            br: rupture_list_from_lt_branch(branch, source_types)
            for br, branch in logic_tree_dict.items()
        }


def rupture_list_from_lt_branch(branch: dict,
                                source_types: Sequence[str] = ('simple_fault')
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
            rups = [
                process_rup(r, source) for source in sources
                for r in source.iter_ruptures()
            ]

            rupture_list.extend(rups)

    return rupture_list


def _process_rup(rup, source):
    rup.source = source.source_id
    return rup


def _process_source(source):
    return [_process_rup(r, source) for r in source.iter_ruptures()]


def rupture_list_from_lt_branch_parallel(
        branch: dict,
        source_types: Sequence[str] = ('simple_fault'),
        n_procs: Optional[int] = None) -> list:
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
        n_procs = _n_procs

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
    Creates a GeoPandas GeoDataFrame from a rupture list.

    :param rupture_list:
        List of :class:`Rupture`s. 

    :returns:
        GeoDataFrame, with two columns, `rupture` which holds the
        :class:`Rupture` object, and `geometry` which has the geometry as a
        Shapely :class:`Point` object.
    """

    df = pd.DataFrame(index=range(len(rupture_list)),
                      data=rupture_list,
                      columns=['rupture'])

    df['geometry'] = df.apply(lambda z: Point(z.rupture.hypocenter.longitude, z
                                              .rupture.hypocenter.latitude),
                              axis=1)
    rupture_gdf = gpd.GeoDataFrame(df)
    rupture_gdf.crs = {'init': 'epsg:4326'}
    return rupture_gdf


def add_ruptures_to_bins(rupture_gdf: gpd.GeoDataFrame,
                         bin_df: gpd.GeoDataFrame,
                         parallel: bool = False) -> None:
    """
    Takes a GeoPandas GeoDataFrame of ruptures and adds them to the ruptures
    list that is an attribute of each :class:`SpacemagBin` based on location and
    magnitude. The spatial binning is performed through a left join via
    GeoPandas, and should use RTree if available for speed. This function
    modifies both GeoDataFrames in memory and does not return any value.

    :param rupture_gdf:
        GeoDataFrame of ruptures; this should have two columns, one of them
        being the `rupture` column with the :class:`Rupture` object, and the
        other being the `geometry` column, with a GeoPandas/Shapely geometry
        class.

    :param bin_df:
        GeoDataFrame of the bins. This should have a `geometry` column with a
        GeoPandas/Shapely geometry and a `SpacemagBin` column that has a
        :class:`SpacemagBin` object.

    :param parallel:

        Boolean flag to perform the magnitude binning of the earthquakes in
        parallel. Currently not implemented.

    :Returns:
        `None`.
    """

    join_df = gpd.sjoin(rupture_gdf, bin_df, how='left')

    rupture_gdf['bin_id'] = join_df['index_right']

    def bin_row(row):
        if not np.isnan(row.bin_id):
            spacemag_bin = bin_df.loc[row.bin_id, 'SpacemagBin']
            nearest_bc = _nearest_bin(row.rupture.mag,
                                      spacemag_bin.mag_bin_centers)
            spacemag_bin.mag_bins[nearest_bc].ruptures.append(row.rupture)

    if parallel is False:
        _ = rupture_gdf.apply(bin_row, axis=1)
        return

    if parallel is True:

        def bin_row_apply(df):
            _ = df.apply(bin_row, axis=1)

        raise NotImplementedError


def _parse_eq_time(
        eq,
        time_cols: Union[List[str], Tuple[str], str, None] = None,
) -> datetime.datetime:
    """
    Parses time information into a :class:`datetime.datetime` time.
    """
    if time_cols is None:
        # warn
        return None

    elif isinstance(time_cols, str):
        time_string = eq[time_cols]

    elif len(time_cols) == 1:
        time_string = eq[time_cols[0]]

    else:
        time_string = str()
        for i, tc in enumerate(time_cols):
            if i < 2:
                time_string += str(eq[tc]) + "-"
            elif i == 2:
                time_string += str(eq[tc]) + " "
            else:
                time_string += str(eq[tc]) + ":"

        time_string = time_string[:-1]

    return dateutil.parser.parse(time_string)


def make_earthquake_gdf_from_csv(
        eq_csv: str,
        x_col: str = 'longitude',
        y_col: str = 'latitude',
        depth: str = 'depth',
        magnitude: str = 'magnitude',
        time: Union[List[str], Tuple[str], str, None] = None,
        source: Optional[str] = None,
        event_id: Optional[str] = None,
        epsg: int = 4326,
) -> gpd.GeoDataFrame:
    """
    Reads an earthquake catalog from a CSV file and returns a GeoDataFrame. The
    required columns are x and y coordinates, depth and magnitude; the time,
    source (i.e. the agency or catalog source for the earthquake data), and an
    event_id are optional. The coordinate system, as an EPSG code, is also
    required; this defaults to 4326 (WGS84) if not given.

    :param eq_csv: file path to CSV

    :param x_col: Name of column with the x coordinate.

    :param y_col: Name of column with the y coordinate.

    :param depth: Name of column with the depth values.

    :param magnitude: Name of column with the magnitude values.

    :param time: Name of column(s) with time values. If multiple values are
        used, they should be arranged in increasing resolution, i.e. year, then
        month, then day, then hour, etc. These will be parsed using `dateutil`
        if possible.  This parsing is brittle and will probably fail with
        multiple columns; it's better to make a single, unambiguously formatted
        column first.

    :param source: Optional column specifying the source of that earthquake.

    :param event_id: Optional columns specifying an event_id for the earthquake.
        It's helpful if it's a unique value, of course, but this isn't required
        at this step.

    :param epsg: EPSG string specifying the coordinate system. Defaults to 4326
        or WGS84.

    :returns: GeoDataFrame of earthquakes, converted to EPSG:4326 (WGS84).
    """

    df = pd.read_csv(eq_csv)

    if time is not None:
        df['datetime'] = df.apply(_parse_eq_time, time_cols=time, axis=1)

    if source is not None:
        df.rename({'source': source})

    if magnitude is not None:
        df.rename({'magnitude': magnitude})

    if event_id is not None:
        df.rename({'event_id': event_id})

    def parse_geometry(row, x=x_col, y=y_col, z=depth):
        return Point(row[x], row[y], row[z])

    df['geometry'] = df.apply(parse_geometry, axis=1)
    df.drop([x_col, y_col], axis=1)

    eq_gdf = gpd.GeoDataFrame(df)
    eq_gdf.crs = {'init': 'epsg:{}'.format(epsg)}

    if epsg is not 4326:
        eq_gdf = eq_gdf.to_crs(epsg=4326)

    eq_gdf['longitude'] = [
        eq['geometry'].xy[0][0] for i, eq in eq_gdf.iterrows()
    ]
    eq_gdf['latitude'] = [
        eq['geometry'].xy[1][0] for i, eq in eq_gdf.iterrows()
    ]

    return eq_gdf


@attr.s(auto_attribs=True)
class Earthquake:
    magnitude: Optional[float] = None
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    depth: Optional[float] = None
    time: Optional[datetime.datetime] = None
    source: Optional[str] = None
    event_id: Optional[Union[float, str, int]] = None


def _make_earthquake_from_row(row):

    eq_args = [
        'magnitude', 'longitude', 'latitude', 'depth', 'time', 'source',
        'event_id'
    ]

    eq_d = {}

    for arg in eq_args:
        try:
            eq_d[arg] = row[arg]
        except KeyError:
            eq_d[arg] = None

    return Earthquake(**eq_d)


def _nearest_bin(val, bin_centers):
    bca = np.array(bin_centers)
    return bin_centers[np.argmin(np.abs(val - bca))]


def add_earthquakes_to_bins(earthquake_gdf: gpd.GeoDataFrame,
                            bin_df: gpd.GeoDataFrame) -> None:
    """
    Takes a GeoPandas GeoDataFrame of observed earthquakes (i.e., an
    instrumental earthquake catalog) and adds them to the ruptures
    list that is an attribute of each :class:`SpacemagBin` based on location and
    magnitude. The spatial binning is performed through a left join via
    GeoPandas, and should use RTree if available for speed. This function
    modifies both GeoDataFrames in memory and does not return any value.

    :param rupture_gdf:
        GeoDataFrame of ruptures; this should have two columns, one of them
        being the `rupture` column with the :class:`Rupture` object, and the
        other being the `geometry` column, with a GeoPandas/Shapely geometry
        class.

    :param bin_df:
        GeoDataFrame of the bins. This should have a `geometry` column with a
        GeoPandas/Shapely geometry and a `SpacemagBin` column that has a
        :class:`SpacemagBin` object.

    :Returns:
        `None`.
    """

    earthquake_gdf['Eq'] = earthquake_gdf.apply(_make_earthquake_from_row,
                                                axis=1)

    join_df = gpd.sjoin(earthquake_gdf, bin_df, how='left')

    earthquake_gdf['bin_id'] = join_df['index_right']

    for i, eq in earthquake_gdf.iterrows():
        if not np.isnan(eq.bin_id):
            spacemag_bin = bin_df.loc[eq.bin_id, 'SpacemagBin']

            if eq.mag < spacemag_bin.min_mag - spacemag_bin.bin_width / 2:
                pass
            elif eq.mag > spacemag_bin.max_mag + spacemag_bin.bin_width / 2:
                pass
            else:
                nearest_bc = _nearest_bin(eq.Eq.mag,
                                          spacemag_bin.mag_bin_centers)
                spacemag_bin.mag_bins[nearest_bc].observed_earthquakes.append(
                    eq['Eq'])
                spacemag_bin.observed_earthquakes[nearest_bc].append(eq['Eq'])


def make_SpacemagBins_from_bin_gis_file(bin_filepath: str,
                                        min_mag: Optional[float] = 6.,
                                        max_mag: Optional[float] = 9.,
                                        bin_width: Optional[float] = 0.2
                                        ) -> gpd.GeoDataFrame:
    """
    Creates a GeoPandas GeoDataFrame with :class:`SpacemagBins` that forms the
    basis of most of the spatial hazard model testing.

    :param bin_filepath:
        Path to GIS polygon file that contains the spatial bins for analysis.

    :param min_mag:
        Minimum earthquake magnitude for MFD-based analysis.

    :param max_mag:
        Maximum earthquake magnitude for MFD-based analysis.

    :param bin_width:
        Width of earthquake/MFD bins.

    :returns:
        GeoDataFrame with :class:`SpacemagBin`s as a column.
    """

    bin_df = gpd.read_file(bin_filepath)

    def bin_to_mag(row):
        return SpacemagBin(row.geometry,
                           bin_id=row._name,
                           min_mag=min_mag,
                           max_mag=max_mag)

    bin_df['SpacemagBin'] = bin_df.apply(bin_to_mag, axis=1)

    return bin_df


def get_source_bins(bin_gdf):
    source_list = []
    for i, row in bin_gdf.iterrows():
        cum_mfd = row.SpacemagBin.get_rupture_mfd(cumulative=True)
        if sum(cum_mfd.values()) > 0:
            source_list.append(i)

    source_bin_gdf = bin_gdf.loc[source_list]
    return source_bin_gdf


def sample_earthquakes(rupture: Union[ParametricProbabilisticRupture,
                                      NonParametricProbabilisticRupture],
                       interval_length: float,
                       t0: float = 0.,
                       rand_seed: Optional[int] = None) -> List[Earthquake]:
    """
    Creates a random sample (in time) of earthquakes from a single rupture.
    Currently only uniformly random (Poissonian) earthquakes are supported.
    Other than the event time, the generated earthquakes should be identical.

    :param rupture:
        Rupture from which the earthquakes will be generated.

    :param interval_length:
        Length of time over which the earthquakes will be sampled.

    :param t0:
        Start time of analysis (in years).  No real need to change this.

    :param rand_seed:
        Seed for random time generation.

    :returns:
        List of :class:`Earthquake`s.
    """

    event_times = sample_event_times_in_interval(rupture.occurrence_rate,
                                                 interval_length, t0,
                                                 rand_seed)
    try:
        source = rupture.source
    except:
        source = None

    eqs = [
        Earthquake(magnitude=rupture.mag,
                   latitude=rupture.hypocenter.latitude,
                   longitude=rupture.hypocenter.longitude,
                   depth=rupture.hypocenter.depth,
                   source=source,
                   time=et) for et in event_times
    ]
    return eqs
