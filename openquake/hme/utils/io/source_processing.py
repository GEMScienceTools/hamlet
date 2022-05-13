import logging
from time import sleep
from functools import partial
from typing import Union, Optional

from h3 import h3
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm.autonotebook import tqdm


from openquake.hazardlib.source.rupture import (
    NonParametricProbabilisticRupture,
    ParametricProbabilisticRupture,
)


from ..utils import (
    Pool,
    logger,
    _n_procs,
    flatten_list,
    _chunk_source_list,
    get_nonparametric_rupture_occurrence_rate,
)


def _process_rupture(
    rup: Union[
        ParametricProbabilisticRupture, NonParametricProbabilisticRupture
    ],
    h3_res: int = 3,
):

    rd = {}

    if isinstance(rup, NonParametricProbabilisticRupture):
        rd["occurrence_rate"] = get_nonparametric_rupture_occurrence_rate(
            rup
        )  # * source_weight
    else:
        rd["occurrence_rate"] = rup.occurrence_rate  # * source_weight

    rd["longitude"] = rup.hypocenter.x
    rd["latitude"] = rup.hypocenter.y
    rd["depth"] = rup.hypocenter.z
    rd["mag"] = rup.mag
    rd["strike"] = rup.surface.get_strike()
    rd["dip"] = rup.surface.get_dip()
    rd["rake"] = rup.rake
    rd["cell_id"] = h3.geo_to_h3(rd["latitude"], rd["longitude"], h3_res)

    return rd


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
            _process_source(source, h3_res=sc["h3_res"], pbar=pbar)
            for source in sc["source_chunk"]
        ],
    )

    del sc["source_chunk"]

    rups = flatten_list(rups)

    return rups


def _process_source(source, h3_res: int = 3, pbar: tqdm = None):

    rups = list(
        map(partial(_process_rupture, h3_res=h3_res), source.iter_ruptures())
    )
    rup_df = pd.DataFrame(rups)
    rup_df.index = [
        "{}_{}".format(source.source_id, i) for i, rup in enumerate(rups)
    ]
    rup_df.index.name = "rup_id"
    if hasattr(source, "weight"):
        rup_df["occurrence_rate"] *= source.weight

    if pbar is not None:
        pbar.update(n=len(rups))

    return rup_df


def rupture_df_from_source_list(source_list, h3_res=3):
    rup_counts = [s.count_ruptures() for s in source_list]
    n_rups = sum(rup_counts)
    # rupture_df = pd.DataFrame(index)
    source_df_list = []
    pbar = tqdm(total=n_rups)

    logging.info("{} ruptures".format(n_rups))

    for i, source in enumerate(source_list):
        rups = list(
            tqdm(
                map(
                    partial(_process_rupture, h3_res=h3_res),
                    source.iter_ruptures(),
                ),
                total=rup_counts[i],
                leave=False,
            )
        )
        rup_df = pd.DataFrame(rups)
        rup_df.index = [
            "{}_{}".format(source.source_id, i) for i, rup in enumerate(rups)
        ]
        rup_df.index.name = "rup_id"
        if hasattr(source, "weight"):
            rup_df["occurrence_rate"] *= source.weight
        source_df_list.append(rup_df)

    rupture_df = pd.concat(source_df_list, axis=0)

    return rupture_df


def rupture_list_from_source_list_parallel(
    source_list, n_procs: int = _n_procs, h3_res: int = 3
) -> pd.DataFrame:

    logger.info("    chunking sources")
    source_chunks, chunk_sums = _chunk_source_list(source_list, n_procs)

    if n_procs > len(source_chunks):
        logging.info("    fewer chunks than processes.")

    logger.info("    beginning multiprocess source processing")
    with Pool(n_procs, maxtasksperchild=None) as pool:

        rupture_dfs = []

        chunks_with_args = [
            {
                "source_chunk": source_chunk,
                "position": i,
                "chunk_sum": chunk_sums[i],
                "h3_res": h3_res,
            }
            for i, source_chunk in enumerate(source_chunks)
        ]

        pbar = tqdm([n for n in range(n_procs)])

        for rups in pool.imap_unordered(
            _process_source_chunk, chunks_with_args
        ):
            rupture_dfs.extend(rups)
            del rups

        pbar.write("\n" * n_procs)
        pbar.close()
        logging.info(
            "    finishing multiprocess source processing, cleaning up."
        )

    while isinstance(rupture_dfs[0], list):
        rupture_dfs = flatten_list(rupture_dfs)

    rupture_df = pd.concat(rupture_dfs, axis=0)
    return rupture_df


def rupture_dict_from_logic_tree_dict(
    logic_tree_dict: dict,
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
        for i, (branch_name, source_list) in enumerate(logic_tree_dict.items()):
            logging.info(
                f"processing {branch_name} ({i+1}/{len(logic_tree_dict.keys())})"
            )
            rup_dict[branch_name] = rupture_list_from_source_list_parallel(
                source_list, h3_res=h3_res, n_procs=n_procs
            )
    else:
        rup_dict = {}
        for i, (branch_name, source_list) in enumerate(logic_tree_dict.items()):
            logging.info(
                f"processing {branch_name} ({i+1}/{len(logic_tree_dict.keys())})"
            )
            rup_dict[branch_name] = rupture_df_from_source_list(
                source_list, h3_res=h3_res
            )
    return rup_dict


def rupture_dict_to_gdf(
    rupture_dict: dict,
    weights: dict,
    return_gdf: bool = False,
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:

    dfs = []

    for branch, branch_df in rupture_dict.items():
        branch_df["occurrence_rate"] *= weights[branch]

        dfs.append(branch_df)

    if len(dfs) == 1:
        df = dfs[0]
    else:
        df = pd.concat(dfs, axis=0)

    if return_gdf:

        def parse_geometry(row, x="longitude", y="latitude", z="depth"):
            return Point(row[x], row[y], row[z])

        df["geometry"] = df.apply(parse_geometry, axis=1)

    return df
