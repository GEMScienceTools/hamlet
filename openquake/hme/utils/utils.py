import os
import json
import logging
import datetime
from time import time, sleep
from functools import partial
from multiprocessing import Pool
from collections import deque
from collections.abc import Mapping
from typing import Sequence, List, Optional, Union, Tuple

import attr
import dateutil
import numpy as np
import pandas as pd
import geopandas as gpd

from h3 import h3
from tqdm import tqdm, trange
from shapely.geometry import Point, Polygon
from openquake.hazardlib.geo.point import Point as OQPoint
from openquake.hazardlib.source.rupture_collection import split
from openquake.hazardlib.source import MultiPointSource, ComplexFaultSource
from openquake.hazardlib.source.rupture import (
    NonParametricProbabilisticRupture,
    ParametricProbabilisticRupture,
)

from .simple_rupture import SimpleRupture
from .bins import SpacemagBin
from .stats import sample_event_times_in_interval

_n_procs = max(1, os.cpu_count() - 1)
# _n_procs = 2  # parallel testing


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


def rupture_dict_from_logic_tree_dict(
    logic_tree_dict: dict,
    simple_ruptures: bool = True,
    parallel: bool = True,
    n_procs: Optional[int] = _n_procs,
) -> dict:
    """
    Creates a dictionary of ruptures from a dictionary representation of a 
    logic tree (as produced by
    :func:`~hztest.utils.io.process_source_logic_tree`). Each branch in the
    logic tree dict is a value (with the branch name as a key) and this
    structure is retained in the resulting rupture dict. However all of the
    ruptures from each source within a branch will be flattened to a single
    list.

    Use the `source_types` argument to specify which types of sources should be
    used to collect ruptures.

    :param logic_tree_dict:
        Seismic source logic tree

    :param simple_ruptures:
        Whether to use
        :class:`openquake.hme.utils.simple_rupture.simple_rupture` to represent
        ruptures, instead of the full OpenQuake version.

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
            branch_name: rupture_list_from_source_list_parallel(
                source_list, simple_ruptures=simple_ruptures, n_procs=n_procs
            )
            for branch_name, source_list in logic_tree_dict.items()
        }
    else:
        return {
            branch_name: rupture_list_from_source_list(
                source_list, simple_ruptures=simple_ruptures
            )
            for branch_name, source_list in logic_tree_dict.items()
        }


def rup_to_dict(rup: SimpleRupture):
    return {
        "strike": rup.strike,
        "dip": rup.dip,
        "rake": rup.rake,
        "mag": rup.mag,
        "lon": rup.hypocenter.x,
        "lat": rup.hypocenter.y,
        "depth": rup.hypocenter.z,
        "occurrence_rate": rup.occurrence_rate,
        "source": rup.source,
    }


def _rupture_from_df_row(row):
    rup = SimpleRupture(
        strike=row["strike"],
        dip=row["dip"],
        rake=row["rake"],
        mag=row["mag"],
        hypocenter=OQPoint(row["lon"], row["lat"], row["depth"]),
        occurrence_rate=row["occurrence_rate"],
        source=row["source"],
    )
    return rup


def _rupture_from_namedtuple(row):
    rup = SimpleRupture(
        strike=row.strike,
        dip=row.dip,
        rake=row.rake,
        mag=row.mag,
        hypocenter=OQPoint(row.lon, row.lat, row.depth),
        occurrence_rate=row.occurrence_rate,
        source=row.source,
    )
    return rup


def _process_ruptures_from_df(rup_df: pd.DataFrame):
    rup_list = list(
        tqdm(
            map(_rupture_from_namedtuple, rup_df.itertuples(index=False, name="rup")),
            total=len(rup_df),
        )
    )
    rupture_df = rupture_list_to_gdf(rup_list)
    return rupture_df


def _process_ruptures_from_df_parallel(rup_df: pd.DataFrame):
    raise NotImplementedError


def read_ruptures_from_dataframe(rup_df):
    new_rup_df = _process_ruptures_from_df(rup_df)
    return new_rup_df


def rupdf_from_dict(rup: dict):
    rupture_list = []
    for (i, r) in rup.iterrows():
        rupture_list.append(
            SimpleRupture(
                strike=r.strike,
                dip=r.dip,
                rake=r["rake"],
                mag=r["mag"],
                hypocenter=OQPoint(r["lon"], r["lat"], r["depth"]),
                occurrence_rate=r["occurrence_rate"],
                source=r.source,
            )
        )

    df = pd.DataFrame(
        index=range(len(rupture_list)), data=rupture_list, columns=["rupture"]
    )
    return df


def rupture_list_from_source_list(
    source_list: list, simple_ruptures: bool = True
) -> list:
    """
    Creates a list of ruptures from all of the sources within a single logic
    tree branch, adding the `source_id` of each source to the rupture as an
    attribute called `source`.

    :param source_list:
        List of sources containing ruptures.

    :param simple_ruptures:
        Whether to use
        :class:`openquake.hme.utils.simple_rupture.SimpleRupture` to represent
        ruptures, instead of the full OpenQuake version.

    :returns:
        All of the ruptures from all sources of `sources_types` in the logic
        tree branch.
    """
    rup_counts = [s.count_ruptures() for s in source_list]
    n_rups = sum(rup_counts)
    rupture_list = []
    pbar = tqdm(total=n_rups)

    logging.info("{} ruptures".format(n_rups))

    for i, source in enumerate(source_list):
        rups = list(tqdm(
                map(
                    partial(_process_rup, source=source,
                            simple_ruptures=simple_ruptures),
                    source.iter_ruptures()),
                total=rup_counts[i], leave=False))
        pbar.update(n=rup_counts[i])
        rupture_list.extend(rups)

    pbar.close()

    return rupture_list


def _process_rup(
    rup: Union[ParametricProbabilisticRupture, NonParametricProbabilisticRupture],
    source,
    simple_ruptures=True,
):

    try:
        if simple_ruptures is False:
            rup.source = source.source_id
        elif isinstance(rup, ParametricProbabilisticRupture):
            rup = SimpleRupture(
                strike=rup.surface.get_strike(),
                dip=rup.surface.get_dip(),
                rake=rup.rake,
                mag=rup.mag,
                hypocenter=rup.hypocenter,
                occurrence_rate=rup.occurrence_rate,
                source=source.source_id,
            )

        elif isinstance(rup, NonParametricProbabilisticRupture):
            occurrence_rate = sum(
                i * prob_occur for i, prob_occur in enumerate(rup.probs_occur)
            )
            rup = SimpleRupture(
                strike=rup.surface.get_strike(),
                dip=rup.surface.get_dip(),
                rake=rup.rake,
                mag=rup.mag,
                hypocenter=rup.hypocenter,
                occurrence_rate=occurrence_rate,
            )
        return rup
    except:
        rup.source = source.source_id
        return rup


def _process_source(source, simple_ruptures: bool = True, pbar: tqdm = None):

    ruptures = list(
        map(
            partial(_process_rup, source=source, simple_ruptures=simple_ruptures),
            source.iter_ruptures(),
        )
    )

    if pbar is not None:
        pbar.update(n=len(ruptures))

    return ruptures


def _process_source_chunk(source_chunk_w_args) -> list:

    sc = source_chunk_w_args

    pos = str(sc["position"] + 1)
    if len(pos) == 1:
        pos = f"0{pos}"

    text = f"source chunk #{pos}"

    pbar = tqdm(total=sc["chunk_sum"], position=sc["position"], desc=text)

    # wait so that processes don't finish at same time and take too much RAM
    sleep(sc["position"] * 0.5)

    rups = (
        [
            _process_source(source, simple_ruptures=sc["simple_ruptures"], pbar=pbar)
            for source in sc["source_chunk"]
        ],
    )

    del sc["source_chunk"]

    rups = flatten_list(rups)

    return rups


def _chunk_source_list(sources: list, n_chunks: int = _n_procs) -> Tuple[list, list]:

    sources_temp = []
    for s in sources:
        if isinstance(s, MultiPointSource):
            for ps in s:
                sources_temp.append(ps)

        # very slow
        # elif isinstance(s, ComplexFaultSource):
        #    for chunk in split(s, chunksize=s.count_ruptures() // n_chunks):
        #        sources_temp.append(chunk)

        else:
            sources_temp.append(s)

    sources = sources_temp

    source_counts = [s.count_ruptures() for s in sources]

    sources = [
        s
        for c, s in sorted(
            zip(source_counts, sources), key=lambda pair: pair[0], reverse=True
        )
    ]

    source_counts.sort(reverse=True)

    source_chunks = [list() for i in range(n_chunks)]
    chunk_sums = np.zeros(n_chunks, dtype=int)

    for i, source in enumerate(sources):
        min_bin = np.argmin(chunk_sums)
        source_chunks[min_bin].append(source)
        chunk_sums[min_bin] += source_counts[i]

    logging.info("     chunk_sums:\n{}".format(str(chunk_sums)))

    source_chunks = [
        chunk for i, chunk in enumerate(source_chunks) if chunk_sums[i] != 0
    ]

    return (source_chunks, chunk_sums.tolist())


def rupture_list_from_source_list_parallel(
    source_list: list, simple_ruptures: bool = True, n_procs: Optional[int] = _n_procs
) -> list:
    """
    Creates a list of ruptures from all of the sources within list,
    adding the `source_id` of each source to the rupture as an
    attribute called `source`.

    Works in parallel.

    :param simple_ruptures:
        Whether to use
        :class:`openquake.hme.utils.simple_rupture.simple_rupture` to represent
        ruptures, instead of the full OpenQuake version.

    :param n_procs:
        Number of parallel processes. If `None` is passed, it defaults to
        `os.cpu_count() -1`.

    :returns:
        All of the ruptures from all sources of `sources_types` in the logic
        tree branch.
    """
    logger.info("    chunking sources")
    source_chunks, chunk_sums = _chunk_source_list(source_list, n_procs)

    if n_procs > len(source_chunks):
        logging.info("    fewer chunks than processes.")

    logger.info("    beginning multiprocess source processing")
    with Pool(n_procs, maxtasksperchild=None) as pool:

        rupture_list = []

        chunks_with_args = [
            {
                "source_chunk": source_chunk,
                "position": i,
                "chunk_sum": chunk_sums[i],
                "simple_ruptures": simple_ruptures,
            }
            for i, source_chunk in enumerate(source_chunks)
        ]

        pbar = tqdm([n for n in range(n_procs)])

        for rups in pool.imap_unordered(_process_source_chunk, chunks_with_args):
            rupture_list.extend(rups)
            rups = ""

        pbar.write("\n" * n_procs)
        pbar.close()
        logging.info("    finishing multiprocess source processing, cleaning up.")

    while isinstance(rupture_list[0], list):
        rupture_list = flatten_list(rupture_list)
    return rupture_list


def _add_rupture_geom(df):
    """
    Makes a `Point` object of the the hypocenter of each rupture in the
    dataframe.
    """
    return df.apply(
        lambda z: Point(z.rupture.hypocenter.longitude, z.rupture.hypocenter.latitude),
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
        index=range(len(rupture_list)), data=rupture_list, columns=["rupture"]
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


def _h3_bin_from_rupture(
    rupture: Union[
        SimpleRupture, NonParametricProbabilisticRupture, ParametricProbabilisticRupture
    ],
    h3_res: int = 3,
) -> str:
    """
    Returns the hexadecimal string that is the index of the `h3` spatial bin
    which contains the hypocenter of the `rupture` at the given level of
    resolution (`res`).

    See the documentation for `h3` (https://uber.github.io/h3/) for more
    information.
    """

    return h3.geo_to_h3(
        rupture.hypocenter.latitude, rupture.hypocenter.longitude, h3_res
    )


def make_bin_gdf_from_rupture_gdf(
    rupture_gdf: gpd.GeoDataFrame,
    h3_res: int = 3,
    parallel: bool = True,
    n_procs: Optional[int] = None,
    min_mag: Optional[float] = 6.0,
    max_mag: Optional[float] = 9.0,
    bin_width: Optional[float] = 0.2,
) -> gpd.GeoDataFrame:
    """
    Takes all of the ruptures, finds the `h3` spatial bins for each, and then
    makes a new `GeoDataFrame` of all of the spatial bins, with
    :class:`~openquake.hme.utils.bins.SpacemagBin` initialized for each of the
    bins.

    :param rupture_gdf:
        `GeoDataFrame` with all of the ruptures.

    :param h3_res:
        Resolution for the `h3` bins (spatial cells). See
        https://uber.github.io/h3/#/ for more information. Defaults to "3",
        which seems good for earthquake analysis. Larger numbers are smaller
        grid cells.

    :param parallel:
        Boolean flag determining whether to find the spatial indices of the
        ruptures in parallel or not. The parallel algorithm has a lot of
        overhead and will be slower with a relatively low number of
        ruptures (hundreds of thousands or so, depending on the computer).

    :param n_procs:
        Number of simultaneous processes run if `parallel` is `True`. Defaults
        to os.cpu_count() - 1 for multicore systems.

    :param min_mag:
        Minimum magnitude of the :class:`~openquake.hme.utils.bins.SpacemagBin`

    :param max_mag:
        Maximum magnitude of the :class:`~openquake.hme.utils.bins.SpacemagBin`

    :param bin_width:
        Width of the :class:`~openquake.hme.utils.bins.SpacemagBin` bins in
        magnitude units.

    """

    n_procs = n_procs or _n_procs

    logging.info("starting rupture-bin spatial join")

    if (parallel is False) or (n_procs == 1):
        rupture_gdf["bin_id"] = list(
            tqdm(
                map(
                    partial(_h3_bin_from_rupture, h3_res=h3_res),
                    rupture_gdf["rupture"].values,
                ),
                total=rupture_gdf.shape[0],
            )
        )
    else:
        with Pool(n_procs) as pool:
            rupture_gdf["bin_id"] = tqdm(
                pool.imap(
                    partial(_h3_bin_from_rupture, h3_res=h3_res),
                    rupture_gdf["rupture"].values,
                    chunksize=500,
                ),
                total=rupture_gdf.shape[0],
            )

    logging.info("finished rupture-bin spatial join")

    hex_codes = list(set(rupture_gdf["bin_id"].values))

    polies = [
        Polygon(h3.h3_to_geo_boundary(hex_code, geo_json=True))
        for hex_code in hex_codes
    ]

    bin_gdf = gpd.GeoDataFrame(
        index=hex_codes, geometry=polies, crs={"init": "epsg:4326", "no_defs": True}
    )
    bin_gdf = make_SpacemagBins_from_bin_gdf(
        bin_gdf, min_mag=min_mag, max_mag=max_mag, bin_width=bin_width
    )
    return bin_gdf


def add_ruptures_to_bins(
    rupture_gdf: gpd.GeoDataFrame, bin_gdf: gpd.GeoDataFrame
) -> None:
    """
    Takes a GeoPandas GeoDataFrame of ruptures and adds them to the ruptures
    list that is an attribute of each
    :class:`~openquake.hme.utils.bins.SpacemagBin` based on location and
    magnitude. This function modifies both GeoDataFrames in memory and does not
    return any value.

    :param rupture_gdf: GeoDataFrame of ruptures; this should have two columns,
        one of them being the `rupture` column with the :class:`Rupture` object,
        and the other being the `geometry` column, with a GeoPandas/Shapely
        geometry class.

    :param bin_gdf: GeoDataFrame of the bins. This should have a `geometry`
        column with a GeoPandas/Shapely geometry and a `SpacemagBin` column that
        has a :class:`~openquake.hme.utils.bins.SpacemagBin` object.

    :Returns: `None`.
    """
    logger.info("    adding ruptures to bins")

    bin_edges = bin_gdf.iloc[0].SpacemagBin.get_bin_edges()
    bin_centers = bin_gdf.iloc[0].SpacemagBin.mag_bin_centers

    logging.info("\tgetting mag bin vals")
    rupture_gdf["mag_r"] = pd.cut(
        list(map(lambda r: r.mag, rupture_gdf["rupture"])),
        bin_edges,
        labels=bin_centers,
    )

    rup_groups = rupture_gdf.groupby(["bin_id"])

    pbar = tqdm(total=len(rupture_gdf))

    for (bin_id, rup_group) in rup_groups:
        sbin = bin_gdf.loc[bin_id, "SpacemagBin"]

        mag_groups = rup_group.groupby("mag_r")

        for mag_bin, mag_group in mag_groups:
            sbin.mag_bins[mag_bin].ruptures.extend(mag_group["rupture"].values)
        pbar.update(len(rup_group))

    pbar.close()
    logging.info("\tdone adding ruptures to bins")
    return


def _parse_eq_time(
    eq, time_cols: Union[List[str], Tuple[str], str, None] = None,
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
    eq_gdf.crs = {"init": "epsg:{}".format(epsg)}

    if epsg != 4326:
        eq_gdf = eq_gdf.to_crs(epsg=4326)

    eq_gdf["longitude"] = [eq["geometry"].xy[0][0] for i, eq in eq_gdf.iterrows()]
    eq_gdf["latitude"] = [eq["geometry"].xy[1][0] for i, eq in eq_gdf.iterrows()]

    return eq_gdf


@attr.s(auto_attribs=True)
# @dataclass
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
        "magnitude",
        "longitude",
        "latitude",
        "depth",
        "time",
        "source",
        "event_id",
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


def add_earthquakes_to_bins(
    earthquake_gdf: gpd.GeoDataFrame,
    bin_df: gpd.GeoDataFrame,
    category: str = "observed",
    h3_res: int = 3,
) -> None:
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

    :param category:
        Type of earthquake catalog. Default value is `observed` which is,
        typically, the catalog used to make the model. If a separate testing
        catalog is to be considered, then this is added using the `prospective`
        value.

    :Returns:
        `None`.
    """

    earthquake_gdf["Eq"] = earthquake_gdf.apply(_make_earthquake_from_row, axis=1)
    earthquake_gdf["bin_id"] = [
        h3.geo_to_h3(eq.latitude, eq.longitude, h3_res) for eq in earthquake_gdf.Eq
    ]

    for i, eq in earthquake_gdf.iterrows():
        try:
            spacemag_bin = bin_df.loc[eq.bin_id, "SpacemagBin"]

            if eq.magnitude < spacemag_bin.min_mag - spacemag_bin.bin_width / 2:
                pass
            elif eq.magnitude > (spacemag_bin.max_mag + spacemag_bin.bin_width / 2):
                pass
            else:
                nearest_bc = _nearest_bin(eq.Eq.magnitude, spacemag_bin.mag_bin_centers)

                if category == "observed":
                    spacemag_bin.mag_bins[nearest_bc].observed_earthquakes.append(
                        eq["Eq"]
                    )
                    spacemag_bin.observed_earthquakes[nearest_bc].append(eq["Eq"])
                elif category == "prospective":
                    spacemag_bin.mag_bins[nearest_bc].prospective_earthquakes.append(
                        eq["Eq"]
                    )
                    spacemag_bin.prospective_earthquakes[nearest_bc].append(eq["Eq"])
        except:
            pass


def make_SpacemagBins_from_bin_gis_file(
    bin_filepath: str,
    min_mag: Optional[float] = 6.0,
    max_mag: Optional[float] = 9.0,
    bin_width: Optional[float] = 0.2,
) -> gpd.GeoDataFrame:
    """
    Creates a GeoPandas GeoDataFrame with
    :class:`~openquake.hme.utils.bins.SpacemagBin` that forms the
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
        GeoDataFrame with 
        :class:`~openquake.hme.utils.bins.SpacemagBin` as a column.
    """

    bin_gdf = gpd.read_file(bin_filepath)
    return make_SpacemagBins_from_bin_gdf(
        bin_gdf, min_mag=min_mag, max_mag=max_mag, bin_width=bin_width
    )


def make_SpacemagBins_from_bin_gdf(
    bin_gdf: gpd.GeoDataFrame,
    min_mag: Optional[float] = 6.0,
    max_mag: Optional[float] = 9.0,
    bin_width: Optional[float] = 0.2,
) -> gpd.GeoDataFrame:
    """
    Creates a GeoPandas GeoDataFrame with
    :class:`~openquake.hme.utils.bins.SpacemagBin` that forms the
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
        GeoDataFrame with
        :class:`~openquake.hme.utils.bins.SpacemagBin` as a column.
    """

    def bin_to_mag(row):
        return SpacemagBin(
            row.geometry,
            bin_id=row._name,
            min_mag=min_mag,
            max_mag=max_mag,
            bin_width=bin_width,
        )

    bin_gdf["SpacemagBin"] = bin_gdf.apply(bin_to_mag, axis=1)

    # create serialization functions and add to instantiated GeoDataFrame
    def to_dict():
        out_dict = {i: bin_gdf.loc[i, "SpacemagBin"].to_dict() for i in bin_gdf.index}

        return out_dict

    bin_gdf.to_dict = to_dict

    def to_json(fp):
        def to_serializable(val):
            """
            modified from Hynek (https://hynek.me/articles/serialization/)
            """
            if isinstance(val, datetime.datetime):
                return val.isoformat() + "Z"
            # elif isinstance(val, enum.Enum):
            #    return val.value
            elif attr.has(val.__class__):
                return attr.asdict(val)
            elif isinstance(val, np.integer):
                return int(val)
            elif isinstance(val, Exception):
                return {
                    "error": val.__class__.__name__,
                    "args": val.args,
                }
            return str(val)

        with open(fp, "w") as ff:
            json.dump(bin_gdf.to_dict(), ff, default=to_serializable)

    bin_gdf.to_json = to_json

    return bin_gdf


def get_source_bins(bin_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Returns a subset of all of the spatial bins, where each bin actually
    contains earthquake sources.

    :param bin_gdf:
        GeoDataFrame of bins

    :returns:
        GeoDataFrame of bins with sources
    """
    source_list = []
    for i, row in bin_gdf.iterrows():
        cum_mfd = row.SpacemagBin.get_rupture_mfd(cumulative=True)
        if sum(cum_mfd.values()) > 0:
            source_list.append(i)

    source_bin_gdf = bin_gdf.loc[source_list]
    return source_bin_gdf


def subset_source(
    bin_gdf: gpd.GeoDataFrame, subset_file: str, buffer: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Takes a subset of the source model where each bin intersects (or is within)
    a geographic region. This is useful for testing a regional model's
    performance in a single country, for example.

    Please note that the SpacemagBins are not cut at the intersection, so any
    SpacemagBin intersecting the border will be included completely.

    :param bin_gdf:
        GeoDataFrame of bins

    :param subset_file:
        File name/path for a vector GIS file with geographic feature(s)
        representing the study area. Should be derived from a Polygon or
        MultiPolygon-type GIS file, but Points or Polyline types may work
        too, especially with a buffer.

    :param buffer:
        Spatial buffer (in degrees) to be applied to `sub_gdf` before
        subsetting. This allows for ruptures with hypocenters outside the
        `sub_gdf` areas that may still affect the study area to be included.
        Defaults to zero.
    """
    sub_gdf = gpd.read_file(subset_file)
    sub_gdf.crs = bin_gdf.crs

    if buffer != 0:
        sub_gdf["geometry"] = sub_gdf["geometry"].buffer(buffer)

    sj = gpd.sjoin(bin_gdf, sub_gdf)

    return bin_gdf.loc[sj.index]


def sample_earthquakes(
    rupture: Union[ParametricProbabilisticRupture, NonParametricProbabilisticRupture],
    interval_length: float,
    t0: float = 0.0,
    rand_seed: Optional[int] = None,
) -> List[Earthquake]:
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
        List of :class:`Earthquake`.
    """

    event_times = sample_event_times_in_interval(
        rupture.occurrence_rate, interval_length, t0, rand_seed
    )
    try:
        source = rupture.source
    except AttributeError:
        source = None

    eqs = [
        Earthquake(
            magnitude=rupture.mag,
            latitude=rupture.hypocenter.latitude,
            longitude=rupture.hypocenter.longitude,
            depth=rupture.hypocenter.depth,
            source=source,
            time=et,
        )
        for et in event_times
    ]
    return eqs


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


def get_model_mfd(bin_gdf: gpd.GeoDataFrame, cumulative: bool = False) -> dict:
    mod_mfd = bin_gdf.iloc[0].SpacemagBin.get_rupture_mfd()
    mag_bin_centers = bin_gdf.iloc[0].SpacemagBin.mag_bin_centers

    for i, row in bin_gdf.iloc[1:].iterrows():
        bin_mod_mfd = row.SpacemagBin.get_rupture_mfd()
        for bin_center, rate in bin_mod_mfd.items():
            mod_mfd[bin_center] += rate

    if cumulative is True:
        cum_mfd = {}
        cum_mag = 0.0
        # dict has descending order
        for cb in mag_bin_centers[::-1]:
            cum_mag += mod_mfd[cb]
            cum_mfd[cb] = cum_mag

        # make new dict with ascending order
        mod_mfd = {cb: cum_mfd[cb] for cb in mag_bin_centers}

    return mod_mfd


def get_obs_mfd(
    bin_gdf: gpd.GeoDataFrame,
    t_yrs: float,
    prospective: bool = False,
    cumulative: bool = False,
) -> dict:
    mag_bin_centers = bin_gdf.iloc[0].SpacemagBin.mag_bin_centers

    if prospective is False:
        obs_mfd = bin_gdf.iloc[0].SpacemagBin.get_empirical_mfd(t_yrs=t_yrs)
    else:
        obs_mfd = bin_gdf.iloc[0].SpacemagBin.get_prospective_mfd(t_yrs=t_yrs)

    for i, row in bin_gdf.iloc[1:].iterrows():
        if prospective is False:
            bin_obs_mfd = row.SpacemagBin.get_empirical_mfd(t_yrs=t_yrs)
        else:
            bin_obs_mfd = row.SpacemagBin.get_prospective_mfd(t_yrs=t_yrs)

        for bin_center, rate in bin_obs_mfd.items():
            obs_mfd[bin_center] += rate

    if cumulative is True:
        cum_mfd = {}
        cum_mag = 0.0
        # dict has descending order
        for cb in mag_bin_centers[::-1]:
            cum_mag += obs_mfd[cb]
            cum_mfd[cb] = cum_mag

        # make new dict with ascending order
        obs_mfd = {cb: cum_mfd[cb] for cb in mag_bin_centers}

    return obs_mfd


def get_model_annual_eq_rate(bin_gdf: gpd.GeoDataFrame) -> float:
    annual_rup_rate = 0.0
    for i, row in bin_gdf.iterrows():
        sb = row.SpacemagBin
        min_bin_center = np.min(sb.mag_bin_centers)
        bin_mfd = sb.get_rupture_mfd(cumulative=True)
        annual_rup_rate += bin_mfd[min_bin_center]

    return annual_rup_rate


def get_total_obs_eqs(bin_gdf: gpd.GeoDataFrame, prospective: bool = False) -> list:
    """
    Returns a list of all of the observed earthquakes within the model domain.
    """
    obs_eqs = []

    for i, row in bin_gdf.iterrows():
        sb = row.SpacemagBin

        if prospective is False:
            for mb in sb.observed_earthquakes.values():
                obs_eqs.extend(mb)
        else:
            for mb in sb.prospective_earthquakes.values():
                obs_eqs.extend(mb)

    return obs_eqs
