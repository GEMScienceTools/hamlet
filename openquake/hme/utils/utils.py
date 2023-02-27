from multiprocessing.dummy import current_process
import os
import json
import logging
import datetime
from time import sleep
from functools import partial
from multiprocessing import Pool
from collections.abc import Mapping
from typing import Sequence, List, Optional, Union, Tuple, Dict

import attr
import dateutil
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import poisson

from h3 import h3

from tqdm.autonotebook import tqdm
from shapely.geometry import Point
from openquake.hazardlib.geo.point import Point as OQPoint

from openquake.hazardlib.source.rupture import (
    NonParametricProbabilisticRupture,
    ParametricProbabilisticRupture,
)


try:
    from .numba_stat_funcs import poisson_sample_vec
except ImportError:
    poisson_sample_vec = np.random.poisson

_n_procs = max(1, os.cpu_count() - 1)
# _n_procs = 2  # parallel testing
os.environ["NUMEXPR_MAX_THREADS"] = str(_n_procs)


class TqdmLoggingHandler(logging.Handler):
    """
    Class to help `tqdm` log to both a log file and print to the screen.
    """

    def __init__(self, level=logging.NOTSET):
        super().__init__(level=level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


# config logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.addHandler(TqdmLoggingHandler())


def parallelize(
    data, func, cores: int = _n_procs, partitions: int = _n_procs * 10, **kwargs
):
    """
    Function to execute the function `func` in parallel over the collection of
    `data` using the Python multiprocessing (`imap`) functionality.  The
    function splits the data into 10 times the number of (virtual) cores by
    default (though this can be changed), runs the `func` over each of those
    partitions, and then reassembles it. This assumes the data was (or will be)
    in a Pandas dataframe.

    :param data: Collection of data which will be the argument to the function
        `func`. Should be a collection that is `numpy`-suitable such as an
        array, list, or Pandas dataframe.

    :param func: Function object that will be run over the data.

    :cores: Number of virtual cores (i.e., simultaneous processes) to be run.
        Defaults to the number of cores detected by `os.cpu_count()` minus 1,
        unless the system only has 1 core, and then 1 core will be used.

    :param partitions:
        Number of chunks that the data should be split into.  Defaults to 10
        times the number of virtual cores (minus 1) on the system.

    :param kwargs:
        Keyword arguments to be passed to `func`.

    :returns:
        Pandas DataFrame with all of the results from `func`.
    """

    data_split = np.array_split(data, partitions)
    pool = Pool(cores)
    result = pd.concat(pool.imap(partial(func, **kwargs), data_split))
    pool.close()
    pool.join()
    return result


def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source


def flatten_list(lol: List[list]) -> list:
    """
    Flattens a list of lists (lol).

    >>> flatten_list([['l'], ['o', 'l']])
    ['l', 'o', 'l']

    """
    return [item for sublist in lol for item in sublist]


def get_nonparametric_rupture_occurrence_rate(
    rup: NonParametricProbabilisticRupture,
) -> float:
    occurrence_rate = sum(
        i * prob_occur for i, prob_occur in enumerate(rup.probs_occur)
    )
    return occurrence_rate


def _add_rupture_geom(df):
    """
    Makes a `Point` object of the the hypocenter of each rupture in the
    dataframe.
    """
    return df.apply(
        lambda z: Point(
            z.rupture.hypocenter.longitude, z.rupture.hypocenter.latitude
        ),
        axis=1,
    )


def rupture_list_to_gdf(
    rupture_list: list, gdf: bool = False, parallel: bool = True
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Creates a Pandas DataFrame or GeoPandas GeoDataFrame from a rupture list.

    :param rupture_list:
        List of :class:`~openquake.hme.utils.simple_rupture.SimpleRupture`.

    :param gdf:
        Boolean flag to determine whether output is GeoDataFrame (True)
        or DataFrame (False).

    :returns:
        GeoDataFrame, with two columns, `rupture` which holds the
        :class:`Rupture` object, and `geometry` which has the geometry as a
        Shapely :class:`~shapely.geometry.Point` object.
    """
    df = pd.DataFrame(
        index=range(len(rupture_list)),
        data=pd.Series(rupture_list),
        columns=["rupture"],
    )

    if gdf is True:
        if parallel is True and _n_procs > 1:
            df["geometry"] = parallelize(df, _add_rupture_geom)
        else:
            df["geometry"] = _add_rupture_geom(df)

        rupture_gdf = gpd.GeoDataFrame(df)
        rupture_gdf.crs = {"init": "epsg:4326", "no_defs": True}
        return rupture_gdf
    else:
        return df


def scale_rup_rate(rup, rate_scale: float):
    rup.occurrence_rate *= rate_scale


def _parse_eq_time(
    eq,
    time_cols: Union[List[str], Tuple[str], str, None] = None,
) -> Union[datetime.datetime, None]:
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
                time_string += str(int(eq[tc])) + "-"
            elif i == 2:
                time_string += str(int(eq[tc])) + " "
            else:
                time_string += str(eq[tc]) + ":"

        time_string = time_string[:-1]

    return dateutil.parser.parse(time_string)


def make_earthquake_gdf_from_csv(
    eq_csv: str,
    x_col: str = "longitude",
    y_col: str = "latitude",
    depth: str = "depth",
    magnitude: str = "magnitude",
    time: Union[List[str], Tuple[str], str, None] = None,
    source: Optional[str] = None,
    event_id: Optional[str] = None,
    epsg: int = 4326,
    h3_res: int = 3,
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
        df["time"] = df.apply(_parse_eq_time, time_cols=time, axis=1)

    if source is not None:
        df.rename({source: "source"}, axis=1, inplace=True)

    if magnitude is not None:
        df.rename({magnitude: "magnitude"}, axis=1, inplace=True)

    if event_id is not None:
        df.rename({event_id: "event_id"}, axis=1, inplace=True)

    def parse_geometry(row, x=x_col, y=y_col, z=depth):
        return Point(row[x], row[y], row[z])

    df["geometry"] = df.apply(parse_geometry, axis=1)
    df.drop([x_col, y_col], axis=1)

    eq_gdf = gpd.GeoDataFrame(df)
    eq_gdf.crs = f"EPSG:{epsg}"

    if epsg != 4326:
        eq_gdf = eq_gdf.to_crs(epsg=4326)

    eq_gdf["longitude"] = [
        eq["geometry"].xy[0][0] for i, eq in eq_gdf.iterrows()
    ]
    eq_gdf["latitude"] = [
        eq["geometry"].xy[1][0] for i, eq in eq_gdf.iterrows()
    ]

    eq_gdf["cell_id"] = [
        h3.geo_to_h3(row.latitude, row.longitude, h3_res)
        for i, row in eq_gdf.iterrows()
    ]

    return eq_gdf


def trim_eq_catalog(
    eq_gdf: gpd.GeoDataFrame, start_date=None, stop_date=None, duration=None
) -> gpd.GeoDataFrame:
    if duration is not None:
        duration_delta = dateutil.relativedelta.relativedelta(years=duration)

    if "time" not in eq_gdf.columns:
        raise ValueError("Earthquake catalog has no time information")

    # (start time, stop time)
    if (start_date is not None) and (stop_date is not None):
        logging.info(
            "Trimming EQ catalog from {} to {}".format(start_date, stop_date)
        )
        start_date = pd.to_datetime(start_date)
        stop_date = pd.to_datetime(stop_date)

    elif (start_date is not None) and (duration is not None):
        logging.info(
            "Trimming EQ catalog from {} to {} years".format(
                start_date, duration
            )
        )
        stop_date = start_date + duration_delta
        start_date = pd.to_datetime(start_date)
        stop_date = pd.to_datetime(stop_date)

    elif (stop_date is not None) and (duration is not None):
        start_date = stop_date - duration_delta
        start_date = pd.to_datetime(start_date)
        stop_date = pd.to_datetime(stop_date)
        logging.info(
            "Trimming EQ catalog from {} back {} years".format(
                stop_date, duration
            )
        )

    else:
        err_msg = "Need 2 of (start_date, stop_date, duration) to trim catalog"
        raise ValueError(err_msg)

    out_gdf = eq_gdf.loc[
        (eq_gdf.time > start_date) & (eq_gdf.time <= stop_date)
    ]

    return out_gdf


def mag_to_mo(mag: float, c: float = 9.05):
    """
    Scalar moment [in Nm] from moment magnitude

    :return:
        The computed scalar seismic moment
    """
    return 10 ** (1.5 * mag + c)


def mo_to_mag(mo: float, c: float = 9.05):
    """
    From moment magnitude to scalar moment [in Nm]

    :return:
        The computed magnitude
    """
    return (np.log10(mo) - c) / 1.5


def get_n_eqs_from_mfd(mfd: dict):
    if np.isscalar(list(mfd.values())[1]):
        return sum(mfd.values())
    else:
        return sum(len(val) for val in mfd.values())


def get_mag_bins(min_mag, max_mag, bin_width):
    bc = round(min_mag, 2)
    bcs = [bc]
    while bc <= max_mag:
        bc += bin_width
        bcs.append(round(bc, 2))

    def get_mag_bin(bc, bin_width=bin_width):
        half_width = bin_width / 2.0
        return (round(bc - half_width, 2), round(bc + half_width, 2))

    mag_bins = {bc: get_mag_bin(bc) for bc in bcs}

    return mag_bins


def get_bin_edges_from_mag_bins(mag_bins: dict):
    bin_centers = sorted(mag_bins.keys())

    bin_edges = [mag_bins[bc][0] for bc in bin_centers]
    bin_edges.append(mag_bins[bin_centers[-1]][1])

    return bin_edges


def get_mag_bins_from_cfg(cfg):
    return get_mag_bins(
        cfg["input"]["bins"]["mfd_bin_min"],
        cfg["input"]["bins"]["mfd_bin_max"],
        cfg["input"]["bins"]["mfd_bin_width"],
    )


def get_model_mfd(rdf, mag_bins, cumulative=False, delete_col=True):
    return get_rup_df_mfd(
        rdf, mag_bins, cumulative=cumulative, delete_col=delete_col
    )


def get_rup_df_mfd(rdf, mag_bins, cumulative=False, delete_col=True):
    bin_centers = np.array(sorted(mag_bins.keys()))
    bin_edges = get_bin_edges_from_mag_bins(mag_bins)

    if "mag_bin" not in rdf.columns:
        rdf["mag_bin"] = pd.cut(
            rdf.magnitude,
            bin_edges,
            right=False,
            include_lowest=True,
            labels=bin_centers,
        )

    mag_bin_groups = rdf.groupby("mag_bin")

    mfd = {}

    for bc in bin_centers:
        mfd[bc] = 0.0
        if bc in mag_bin_groups.groups.keys():
            mfd[bc] += rdf.loc[mag_bin_groups.groups[bc]].occurrence_rate.sum()

    if cumulative is True:
        cum_mfd = {}
        cum_mag = 0.0
        # dict has descending order
        for cb in bin_centers[::-1]:
            cum_mag += mfd[cb]
            cum_mfd[cb] = cum_mag

        # makde new dict with ascending order
        mfd = {cb: cum_mfd[cb] for cb in bin_centers}

    if delete_col:
        del rdf["mag_bin"]

    return mfd


def get_obs_mfd(
    rdf, mag_bins, t_yrs: float = 1.0, cumulative=False, delete_col=False
):
    bin_centers = np.array(sorted(mag_bins.keys()))
    bin_edges = get_bin_edges_from_mag_bins(mag_bins)

    if "mag_bin" not in rdf.columns:
        rdf["mag_bin"] = pd.cut(
            rdf.magnitude,
            bin_edges,
            right=False,
            include_lowest=True,
            labels=bin_centers,
        )

    mag_bin_groups = rdf.groupby("mag_bin")

    mfd = {}

    for bc in bin_centers:
        mfd[bc] = 0.0
        if bc in mag_bin_groups.groups.keys():
            mfd[bc] += (
                rdf.loc[mag_bin_groups.groups[bc]].magnitude.count() / t_yrs
            )

    if cumulative is True:
        cum_mfd = {}
        cum_mag = 0.0
        # dict has descending order
        for cb in bin_centers[::-1]:
            cum_mag += mfd[cb]
            cum_mfd[cb] = cum_mag

        # makde new dict with ascending order
        mfd = {cb: cum_mfd[cb] for cb in bin_centers}

    if delete_col:
        del rdf["mag_bin"]

    return mfd


def sample_rups(rup_df, t_yrs, min_mag=1.0, max_mag=10.0):
    mag_idx = (rup_df["magnitude"] >= min_mag) & (
        rup_df["magnitude"] <= max_mag
    )

    rup_rates = rup_df["occurrence_rate"].values * t_yrs
    n_rups = poisson_sample_vec(rup_rates)
    sample_idx = n_rups > 0

    final_idx = sample_idx & mag_idx

    n_samples_per_rup = n_rups[final_idx]
    rup_rows = rup_df.index[final_idx]

    sampled_rups_idx = [
        row
        for i, row in enumerate(rup_rows)
        for j in range(n_samples_per_rup[i])
    ]

    sampled_rups = rup_df.loc[pd.Index(sampled_rups_idx)]

    return sampled_rups


def trim_inputs(input_data, cfg):
    mag_bins = get_mag_bins_from_cfg(cfg)

    min_bin_mag = mag_bins[sorted(mag_bins.keys())[0]][0]
    max_bin_mag = mag_bins[sorted(mag_bins.keys())[-1]][1]

    rup_gdf = input_data["rupture_gdf"]
    eq_gdf = input_data["eq_gdf"]

    mag_range_idxs = (rup_gdf.magnitude >= min_bin_mag) & (
        rup_gdf.magnitude <= max_bin_mag
    )
    input_data["rupture_gdf"] = rup_gdf.loc[mag_range_idxs]
    input_data["cell_groups"] = input_data["rupture_gdf"].groupby("cell_id")

    eq_mag_range_idxs = (eq_gdf.magnitude >= min_bin_mag) & (
        eq_gdf.magnitude <= max_bin_mag
    )

    input_data["eq_gdf"] = eq_gdf.loc[eq_mag_range_idxs]
    input_data["eq_groups"] = input_data["eq_gdf"].groupby("cell_id")

    if "pro_gdf" in input_data.keys():
        pro_gdf = input_data["pro_gdf"]
        pro_mag_range_idxs = (pro_gdf.magnitude >= min_bin_mag) & (
            pro_gdf.magnitude <= max_bin_mag
        )

        input_data["pro_gdf"] = pro_gdf.loc[pro_mag_range_idxs]
        input_data["pro_groups"] = input_data["pro_gdf"].groupby("cell_id")


def get_poisson_counts_from_mfd_iter(mfd: Dict[float, float], n_iters: int):
    samples = {
        mag: np.random.poisson(rate, n_iters) for mag, rate in mfd.items()
    }

    sample_mfds = {
        i: {mag: rate[i] for mag, rate in samples.items()}
        for i in range(n_iters)
    }

    return sample_mfds


def get_cell_rups(cell_id, rupture_gdf, cell_groups):
    cell_rup_df = rupture_gdf.loc[cell_groups.groups[cell_id]]
    return cell_rup_df


def get_cell_eqs(cell_id, eq_gdf, eq_groups):
    if cell_id in eq_groups.groups:
        cell_eqs = eq_gdf.loc[eq_groups.groups[cell_id]]
    else:
        cell_eqs = pd.DataFrame(columns=eq_gdf.columns)
    return cell_eqs


def strike_dip_to_norm_vec(strike, dip):
    strike_rad, dip_rad = np.radians((strike, dip))

    n = np.sin(strike_rad) * np.sin(dip_rad)
    e = -np.cos(strike_rad) * np.sin(dip_rad)
    d = np.cos(dip_rad)

    return np.array([n, e, d])


def angle_between_planes(strike1, dip1, strike2, dip2, return_radians=True):
    nv1 = strike_dip_to_norm_vec(strike1, dip1)
    nv2 = strike_dip_to_norm_vec(strike2, dip2)

    angle = np.arccos(np.dot(nv1, nv2))

    if return_radians == False:
        angle = np.degrees(angle)
    return angle


def angles_between_plane_and_planes(
    strike1, dip1, strikes, dips, return_radians=True
):
    nv1 = strike_dip_to_norm_vec(strike1, dip1)
    nvs = np.array(
        [strike_dip_to_norm_vec(s, dips[i]) for i, s in enumerate(strikes)]
    )

    dots = np.array([nv1.dot(nv) for nv in nvs])
    angles = np.arccos(dots)

    if return_radians == False:
        angles = np.degrees(angles)
    return angles


def angle_between_rakes(rake1, rake2, return_radians=True):
    rake1_rad, rake2_rad = np.radians([rake1, rake2])
    nv1 = np.array([np.cos(rake1_rad), np.sin(rake1_rad)])
    nv2 = np.array([np.cos(rake2_rad), np.sin(rake2_rad)])

    angle = np.arccos(np.dot(nv1, nv2))

    if return_radians == False:
        angle = np.degrees(angle)
    return angle


def angles_between_rake_and_rakes(rake1, rakes, return_radians=True):
    rake1_rad = np.radians(rake1)
    rakes_rad = np.radians(rakes)

    nv1 = np.array([np.cos(rake1_rad), np.sin(rake1_rad)])
    nvs = np.array(
        [np.array([np.cos(rake), np.sin(rake)] for rake in rakes_rad)]
    )

    dots = np.array([nv1.dot(nv) for nv in nvs])
    angles = np.arccos(dots)

    if return_radians == False:
        angles = np.degrees(angles)
    return angles
