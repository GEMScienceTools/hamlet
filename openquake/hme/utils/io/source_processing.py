import logging
import sys
import warnings
from time import sleep
from typing import Union, Optional, Tuple
import multiprocessing

# forkserver avoids inheriting live TBB/tkinter state from the parent while
# still sharing copy-on-write pages with the helper process (Linux only).
# All other platforms use spawn (the only option on Windows, already the
# default on macOS).
if sys.platform == 'linux':
    _mp_ctx = multiprocessing.get_context('forkserver')
else:
    _mp_ctx = multiprocessing.get_context('spawn')

from h3 import h3
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from tqdm.autonotebook import tqdm

warnings.filterwarnings(
    "ignore",
    message="Modules under `h3.unstable` are experimental, and may change at any time.",
)

from h3.unstable import vect as h3_vect  # name in 3.x

from openquake.hazardlib.source.rupture import (
    NonParametricProbabilisticRupture,
    ParametricProbabilisticRupture,
)
from openquake.hazardlib.source import MultiPointSource, ComplexFaultSource


from ..utils import (
    logger,
    _n_procs,
    flatten_list,
    get_nonparametric_rupture_occurrence_rate,
    _get_class_name,
    breakpoint,
)


def _put_sources_in_chunks(sources, source_counts, n_chunks, unweighted=None):
    source_chunks = [list() for i in range(n_chunks)]
    chunk_sums = np.zeros(n_chunks, dtype=int)
    if unweighted is not None:
        uw_chunk_sums = np.zeros(n_chunks, dtype=int)

    for i, source in enumerate(sources):
        min_bin = np.argmin(chunk_sums)
        source_chunks[min_bin].append(source)
        chunk_sums[min_bin] += source_counts[i]
        if unweighted is not None:
            uw_chunk_sums[min_bin] += unweighted[i]

    if unweighted is None:
        return source_chunks, chunk_sums
    else:
        return source_chunks, chunk_sums, uw_chunk_sums


def _chunk_source_list(
    sources: list,
    source_counts_unweighted: list = [],
    n_chunks: int = _n_procs,
    n_rup_threshold=10_000_000,
) -> Tuple[list, list]:
    sources_temp = []
    for s in sources:
        if isinstance(s, MultiPointSource):
            for ps in s:
                sources_temp.append(ps)
        else:
            sources_temp.append(s)

    sources = sources_temp

    def source_weight(source):
        if isinstance(source, ComplexFaultSource):
            weight = 5.0
        else:
            weight = 1.0

        return weight

    if source_counts_unweighted == []:
        logging.info("     no rup counts provided; counting...")
        source_counts_unweighted = [s.count_ruptures() for s in sources]
    else:
        if len(source_counts_unweighted) != len(sources):
            logging.info("    source weights have wrong length, recalculating")
            source_counts_unweighted = [s.count_ruptures() for s in sources]

    source_counts = [
        source_counts_unweighted[i] * source_weight(s)
        for i, s in enumerate(sources)
    ]

    sources = [
        s
        for c, s in sorted(
            zip(source_counts, sources), key=lambda pair: pair[0], reverse=True
        )
    ]

    source_counts_unweighted = [
        s
        for c, s in sorted(
            zip(source_counts, source_counts_unweighted),
            key=lambda pair: pair[0],
            reverse=True,
        )
    ]

    source_counts.sort(reverse=True)

    if sum(source_counts) > n_rup_threshold:
        # n_chunks = n_chunks // 2  # save ram for big models
        pass

    logging.info("     first chunking")
    source_chunks, chunk_sums, uw_chunk_sums = _put_sources_in_chunks(
        sources, source_counts, n_chunks, unweighted=source_counts_unweighted
    )

    nc = n_chunks

    ii = 0
    while ((chunk_sums[-1] * 1.3) < chunk_sums[0]) and (nc > 3):
        if ii == 0:
            logging.info("     rebalancing")
        ii += 1

        nc -= 1
        source_chunks, chunk_sums, uw_chunk_sums = _put_sources_in_chunks(
            sources, source_counts, nc, unweighted=source_counts_unweighted
        )

    logging.info("     chunk_sums:\n{}".format(str(chunk_sums)))

    return (source_chunks, uw_chunk_sums.tolist())


def _process_rupture(
    rup: Union[
        ParametricProbabilisticRupture, NonParametricProbabilisticRupture
    ],
):
    rd = np.zeros(8, dtype=float)  # experiment w/ lower precision later

    if isinstance(rup, NonParametricProbabilisticRupture):
        rd[7] = get_nonparametric_rupture_occurrence_rate(rup)
    else:
        rd[7] = rup.occurrence_rate

    rd[0] = rup.hypocenter.x
    rd[1] = rup.hypocenter.y
    rd[2] = rup.hypocenter.z
    rd[3] = rup.mag
    rd[4] = rup.surface.get_strike()
    rd[5] = rup.surface.get_dip()
    rd[6] = rup.rake

    return rd


def _add_rup_data(i, rup, rup_data):
    rup_data[i, :] = _process_rupture(rup)


def _process_source_chunk(source_chunk_w_args) -> list:
    sc = source_chunk_w_args

    pos = str(sc["position"] + 1)
    if len(pos) == 1:
        pos = f"0{pos}"

    text = f"source chunk #{pos}"

    tqdm.monitor_interval = 0  # prevent monitor thread hang on Linux
    pbar = tqdm(total=sc["chunk_sum"], position=sc["position"], desc=text)

    # wait so that processes don't finish at same time and take too much RAM
    sleep(sc["position"] * 0.5)

    rups = (
        [
            # _process_source(source, h3_res=sc["h3_res"], pbar=None)
            _process_source(
                source,
                h3_res=sc["h3_res"],
                pbar=pbar,
                return_surface=sc["return_surface"],
                return_trt=sc["return_trt"],
            )
            for source in sc["source_chunk"]
        ],
    )

    del sc["source_chunk"]

    rups = flatten_list(rups)

    pbar.close()
    return rups


def _process_source(
    source,
    h3_res: int = 3,
    n_rups: Optional[int] = None,
    pbar: tqdm = None,
    return_surface: bool = False,
    return_trt: bool = False,
):
    rup_cols = [
        "longitude",
        "latitude",
        "depth",
        "magnitude",
        "strike",
        "dip",
        "rake",
        "occurrence_rate",
    ]  # cell_id comes later

    if n_rups is None:
        n_rups = source.count_ruptures()

    if return_surface:
        surfaces = []

    if return_trt:
        trts = []

    rup_data = np.zeros((n_rups, len(rup_cols)), dtype=float)
    cell_ids = []

    for i, rup in enumerate(source.iter_ruptures()):
        _add_rup_data(i, rup, rup_data)
        cell_ids.append(h3.geo_to_h3(rup_data[i, 1], rup_data[i, 0], h3_res))
        if return_surface:
            surfaces.append(rup.surface)
        if return_trt:
            trts.append(rup.tectonic_region_type)
        if pbar is not None:
            pbar.update(n=1)

    rup_df = pd.DataFrame(rup_data, columns=rup_cols)

    rup_df["cell_id"] = pd.Categorical(cell_ids)

    rup_df.index = ["{}_{}".format(source.source_id, i) for i in rup_df.index]
    rup_df.index.name = "rup_id"

    if return_surface:
        rup_df["surface"] = surfaces

    if return_trt:
        rup_df["tectonic_region_type"] = trts

    return rup_df


def rupture_df_from_source_list(
    source_list: list,
    simple_ruptures: bool = False,
    return_trt: bool = False,
    h3_res: int = 3,
):

    if simple_ruptures:
        return_surface = False
    else:
        return_surface = True

    rup_counts = [s.count_ruptures() for s in source_list]
    n_rups = sum(rup_counts)
    source_df_list = []
    pbar = tqdm(total=n_rups)

    logging.info("{} ruptures".format(n_rups))

    for i, source in enumerate(source_list):
        rup_df = _process_source(
            source,
            h3_res=h3_res,
            pbar=pbar,
            return_surface=return_surface,
            return_trt=return_trt,
        )
        source_df_list.append(rup_df)

    rupture_df = pd.concat(source_df_list, axis=0)

    return rupture_df


def rupture_list_from_source_list_parallel(
    source_list: list,
    source_rup_counts: list = [],
    simple_ruptures: bool = False,
    return_trt: bool = False,
    n_procs: int = _n_procs,
    h3_res: int = 3,
) -> pd.DataFrame:

    if simple_ruptures:
        return_surface = False
    else:
        return_surface = True

    logger.info("    chunking sources")
    source_chunks, chunk_sums = _chunk_source_list(
        source_list, source_rup_counts, n_procs
    )

    chunks_with_args = [
        {
            "source_chunk": source_chunk,
            "position": i + 1,
            "chunk_sum": chunk_sums[i],
            "h3_res": h3_res,
            "return_surface": return_surface,
            "return_trt": return_trt,
        }
        for i, source_chunk in enumerate(source_chunks)
    ]

    if n_procs > len(source_chunks):
        logging.info("    fewer chunks than processes.")
        n_procs = len(source_chunks)

    logger.info("    beginning multiprocess source processing")
    pbar = tqdm([n for n in range(n_procs)])

    with _mp_ctx.Pool(n_procs, maxtasksperchild=1) as pool:
        rupture_dfs = []

        chunks_with_args = [
            {
                "source_chunk": source_chunk,
                "position": i,
                "chunk_sum": chunk_sums[i],
                "h3_res": h3_res,
                "return_surface": return_surface,
                "return_trt": return_trt,
            }
            for i, source_chunk in enumerate(source_chunks)
        ]

        for rups in pool.imap_unordered(
            _process_source_chunk, chunks_with_args
        ):
            rupture_dfs.extend(rups)
            del rups

        pbar.write("\n" * n_procs)
        pbar.close()

    logging.info("    finishing multiprocess source processing, cleaning up.")

    while isinstance(rupture_dfs[0], list):
        rupture_dfs = flatten_list(rupture_dfs)

    rupture_df = pd.concat(rupture_dfs, axis=0)
    return rupture_df


def rupture_dict_from_logic_tree_dict(
    logic_tree_dict: dict,
    source_rup_counts: dict,
    simple_ruptures: bool = True,
    return_trt: bool = False,
    parallel: bool = True,
    n_procs: int = _n_procs,
    h3_res: int = 3,
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
        rup_dict = {}
        for i, (branch_name, source_list) in enumerate(
            logic_tree_dict.items()
        ):
            logging.info(
                f"processing {branch_name} ({i+1}/"
                + f"{len(logic_tree_dict.keys())})"
            )
            rup_dict[branch_name] = rupture_list_from_source_list_parallel(
                source_list,
                source_rup_counts=source_rup_counts[branch_name],
                simple_ruptures=simple_ruptures,
                return_trt=return_trt,
                h3_res=h3_res,
                n_procs=n_procs,
            )
    else:
        rup_dict = {}
        for i, (branch_name, source_list) in enumerate(
            logic_tree_dict.items()
        ):
            logging.info(
                f"processing {branch_name} ({i+1}/"
                + f"{len(logic_tree_dict.keys())})"
            )
            rup_dict[branch_name] = rupture_df_from_source_list(
                source_list,
                h3_res=h3_res,
                simple_ruptures=simple_ruptures,
                return_trt=return_trt,
            )
    return rup_dict


def rupture_dict_to_gdf(
    rupture_dict: dict,
    weights: dict,
    trim_zero_rate_rups: bool = True,
    return_gdf: bool = False,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    dfs = []

    for branch, branch_df in rupture_dict.items():
        branch_df = branch_df.copy()
        w = weights[branch]
        if isinstance(w, dict):
            src_ids = branch_df.index.str.rsplit("_", n=1).str[0]
            branch_df["occurrence_rate"] *= src_ids.map(w)
        else:
            branch_df["occurrence_rate"] *= w
        branch_df.index = branch_df.index.values + f"_{branch}"
        branch_df["branch"] = branch

        dfs.append(branch_df)

    if len(dfs) == 1:
        df = dfs[0]
    else:
        df = pd.concat(dfs, axis=0)

    if trim_zero_rate_rups:
        if df.occurrence_rate.min() <= 0.0:
            logging.info("trimming zero-rate ruptures")
            df_len = len(df)
            df = df[df.occurrence_rate > 0.0]
            df_trim_len = len(df)
            logging.info(
                f"{df_trim_len} ruptures remaining, "
                + f"{df_len - df_trim_len} removed"
            )

    if return_gdf:

        def parse_geometry(row, x="longitude", y="latitude", z="depth"):
            return Point(row[x], row[y], row[z])

        df["geometry"] = df.apply(parse_geometry, axis=1)

    logging.info(
        f"Mean annual occurrence rate (pre-trim): {df.occurrence_rate.sum()}"
    )

    return df


def _get_h3_cell(args):
    return h3.geo_to_h3(*args)


def _get_h3_cell_for_rupture_df(rupture_df, h3_res):
    lats = rupture_df["latitude"].to_numpy()
    lons = rupture_df["longitude"].to_numpy()

    cell_ints = h3_vect.geo_to_h3(lats, lons, h3_res)
    h3_to_string = np.frompyfunc(h3.h3_to_string, 1, 1)
    cell_ids = h3_to_string(cell_ints)
    rupture_df["cell_id"] = cell_ids


def make_cell_gdf_from_ruptures(rupture_gdf):
    cell_ids = sorted(rupture_gdf.cell_id.unique())
    polies = [
        Polygon(h3.h3_to_geo_boundary(cell_id, geo_json=True))
        for cell_id in cell_ids
    ]

    cell_gdf = gpd.GeoDataFrame(
        index=cell_ids, geometry=polies, crs="EPSG:4326"
    )

    return cell_gdf
