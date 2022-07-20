import os
import json
import logging

# from types import NoneType
from typing import Union, Optional, Sequence

import pandas as pd
from tqdm import tqdm
from geopandas import GeoDataFrame
from openquake.hazardlib.geo.point import Point


from openquake.commonlib import readinput, logs
from openquake.commonlib.logictree import SourceModelLogicTree
from openquake.hazardlib.source import (
    AreaSource,
    ComplexFaultSource,
    CharacteristicFaultSource,
    NonParametricSeismicSource,
    PointSource,
    MultiPointSource,
    SimpleFaultSource,
    MultiFaultSource,
)


try:
    from openquake.hazardlib.geo.mesh import surface_to_array
except ImportError:
    from openquake.hazardlib.geo.mesh import surface_to_arrays

    def surface_to_array(surface):
        return surface_to_arrays(surface)[0]


def _source_to_series(source):

    if isinstance(source, AreaSource):
        source_type = "area"
    elif isinstance(source, (SimpleFaultSource, CharacteristicFaultSource)):
        source_type = "simple_fault"
    elif isinstance(source, MultiFaultSource):
        source_type = "multi_fault"
    elif isinstance(source, ComplexFaultSource):
        source_type = "complex_fault"
    elif isinstance(source, PointSource):
        source_type = "point"
    elif isinstance(source, MultiPointSource):
        source_type = "multipoint"
    elif isinstance(source, NonParametricSeismicSource):
        source_type = "nonpar"
    else:
        return

    return pd.Series(
        {
            "source": source,
            "tectonic_region_type": source.tectonic_region_type,
            "source_type": source_type,
        }
    )


def sort_sources(
    branch_sources: dict,
    source_types: Optional[Sequence[str]] = None,
    tectonic_region_types: Optional[Sequence[str]] = None,
    branch: Optional[str] = None,
) -> dict:
    """
    Creates lists of sources for each branch of interest, optionally filtering
    sources by `source_type` and `tectonic_region_type`.

    :param source_types:
        Types of sources to collect sources from. Any values in
        (`simple_fault1, `complex_fault`, `area`, `point`, `multipoint`)
        are allowed.  Must be passed as a sequence (i.e., a tuple or list).
        Specify `None` if all values are to be included.

    :param tectonic_region_types:
        Types of tectonic regions to collect sources from. Any values allowed
        in OpenQuake are allowed here. Must be passed as a sequence (i.e., a
        tuple or list).  Specify `None` if all values are to be included.

    :param branch:
        Branch to be evaluated; other branches will be skipped if this is not
        `None`.

    :returns:
        Dictionary with one list of sources per branch considered.
    """

    branch_source_lists = {}

    for branch_name, source_file_list in branch_sources.items():
        if branch_name == branch or branch is None:

            source_list = []

            for source_file in source_file_list:
                try:
                    sources_from_file = read(
                        source_file,
                        get_info=False,
                        area_source_discretization=15.0,
                        rupture_mesh_spacing=2.0,
                        complex_fault_mesh_spacing=5.0,
                    )
                except Exception as e:
                    logging.warning(
                        f"error reading {branch} {source_file}: {e}"
                    )

                try:
                    sources_from_file
                except NameError:
                    break

                for source in sources_from_file:
                    if isinstance(source, list):
                        source_list.extend(
                            [_source_to_series(s) for s in source]
                        )
                    elif source is None:
                        pass
                    else:
                        source_list.append(_source_to_series(source))

            if len(source_list) == 1:
                source_df = source_list[0].to_frame().transpose()
            else:
                source_df = pd.concat(source_list, axis=1).transpose()

            logging.info(f"source df shape:{source_df.shape}")

            if source_types is not None:
                source_df = source_df[source_df.source_type.isin(source_types)]
            if tectonic_region_types is not None:
                source_df = source_df[
                    source_df.tectonic_region_type.isin(tectonic_region_types)
                ]

            branch_source_lists[branch_name] = source_df["source"].to_list()

    return branch_source_lists


def read_branch_sources(
    base_dir, lt_file="ssmLT.xml", branch: Optional[str] = None
):
    lt = SourceModelLogicTree(os.path.join(base_dir, lt_file))

    branch_sources = {}
    weights = {}
    for branch_name, branch_filename in lt.branches.items():
        if branch_name == branch or branch is None:
            try:
                branch_sources[branch_name] = [
                    os.path.join(base_dir, v)
                    for v in branch_filename.value.split()
                ]
                weights[branch_name] = lt.branches[branch_name].weight
            except:
                print("error in ", branch_name)
                pass

    if len(weights.keys()) == 1:
        weights[list(branch_sources.keys())[0]] = 1.0

    logging.info("weights: " + str(weights))

    return branch_sources, weights


def process_source_logic_tree(
    base_dir: str,
    lt_file: str = "ssmLT.xml",
    branch: Optional[str] = None,
    source_types: Optional[Sequence] = None,
    tectonic_region_types: Optional[Sequence] = None,
    verbose: bool = False,
):
    if verbose:
        print("reading source branches")
    branch_sources, weights = read_branch_sources(base_dir, lt_file=lt_file)
    lt = sort_sources(
        branch_sources,
        source_types=source_types,
        tectonic_region_types=tectonic_region_types,
        branch=branch,
    )

    if verbose:
        print(lt.keys())

    if branch != None:
        weights = {branch: 1.0}

    return lt, weights


def process_source_logic_tree_oq(
    job_ini_file,
    base_dir: str,
    lt_file: str = "ssmLT.xml",
    gmm_lt_file: str = "gmmLT.xml",
    sites_file: Optional[str] = None,
    branch: Optional[str] = None,
    source_types: Optional[Sequence] = None,
    tectonic_region_types: Optional[Sequence] = None,
    description: Optional[str] = None,
    verbose: bool = False,
):

    branch_weights = get_branch_weights(base_dir, lt_file)
    logging.info("weights: " + str(branch_weights))
    if job_ini_file is not None:
        job_ini = readinput.get_params(job_ini_file)
    else:
        job_ini = make_job_ini(base_dir, lt_file, gmm_lt_file, description)

    oqp = readinput.get_oqparam(job_ini)

    # only for incomplete models? worried about distance filtering of sources.
    if job_ini_file is None:
        oqp.ground_motion_fields = False

    if branch is not None:
        logging.info(f"reading {branch} (1/1)")
        branch_csms = {
            branch: readinput.get_composite_source_model(oqp, branchID=branch)
        }
        branch_weights = {branch: 1.0}
    else:
        branch_csms = {}
        try:
            # br: readinput.get_composite_source_model(oqp, branchID=br) }
            for i, br in enumerate(branch_weights.keys()):
                logging.info(
                    f"reading {br} ({i+1}/{len(branch_weights.keys())})"
                )
                branch_csms[br] = readinput.get_composite_source_model(
                    oqp, branchID=br
                )
        except Exception as e:
            logging.warn("Can't parse branches in source model... Combining.")
            branch_csms["composite"] = readinput.get_composite_source_model(oqp)
            branch_weights = {"composite": 1.0}

    branch_sources = {}

    for br, branch_csm in branch_csms.items():
        br_sources = []
        for src_group in branch_csm.src_groups:
            if (
                tectonic_region_types is None
                or src_group.trt in tectonic_region_types
            ):
                for src in src_group.sources:
                    if (
                        source_types is None
                        or src.__class__.__name__ in source_types
                    ):
                        br_sources.append(src)
        branch_sources[br] = br_sources

    return branch_sources, branch_weights


def get_branch_weights(base_dir: str, lt_file: str = "ssmLT.xml"):
    ssm_lt_path = os.path.join(base_dir, lt_file)

    lt = SourceModelLogicTree(ssm_lt_path)

    weights = {
        branch_name: lt.branches[branch_name].weight
        for branch_name in lt.branches.keys()
    }

    return weights


def make_job_ini(
    base_dir: str,
    lt_file: str = "ssmLT.xml",
    gmm_lt_file: str = "gmmLT.xml",
    description: Optional[str] = None,
):
    ssm_lt_path = os.path.join(base_dir, lt_file)
    gmm_lt_path = os.path.join(base_dir, gmm_lt_file)
    job_ini = dict(
        calculation_mode="preclassical",
        description=description,
        rupture_mesh_spacing="2.0",
        area_source_discretization="15.0",
        width_of_mfd_bin="0.1",  # typically smaller than from cfg; use cfg?
        reference_vs30_type="measured",
        reference_vs30_value="800.0",
        reference_depth_to_1pt0km_per_sec="30.0",
        maximum_distance="200",
        investigation_time="1.0",
        inputs=dict(
            source_model_logic_tree=ssm_lt_path, gsim_logic_tree=gmm_lt_path
        ),
    )

    return job_ini
