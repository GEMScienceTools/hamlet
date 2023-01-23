import os
import io
import json
import logging
import configparser

# from types import NoneType
from typing import Union, Optional, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm
from geopandas import GeoDataFrame
from openquake.baselib.general import AccumDict
from openquake.calculators.base import run_calc
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


def csm_from_job_ini(job_ini):

    if isinstance(job_ini, dict):
        calc = run_calc(job_ini,
        #calclation_mode="preclassical",
        split_sources="true",
        ground_motion_fields=False,
        )

    else:
        calc = run_calc(
            job_ini,
            calculation_mode="preclassical",
            split_sources="true",
            ground_motion_fields=False,
        )

    sources = calc.csm.get_sources()
    source_info = calc.datastore["source_info"][:]

    for i, source in enumerate(sources):
        source.source_id = i

    return calc.csm, sources, source_info


def get_rlz_source(rlz, csm):
    srcs = []
    grp = csm.get_groups(rlz)
    for g in grp:
        srcs.extend(g)
    return srcs


def get_csm_rlzs(csm):
    csm_rlz_groups = {}
    for i, rlz in enumerate(csm.sm_rlzs):
        csm_rlz_groups[i] = {
            "weight": rlz.weight,
            "sources": get_rlz_source(i, csm),
        }
        return csm_rlz_groups


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

    if job_ini_file is not None:
        job_ini = job_ini_file
    else:
        job_ini = make_job_ini(base_dir, lt_file, gmm_lt_file, description)

    csm, _sources, _source_info = csm_from_job_ini(job_ini)

    logging.info("Realizations:")
    logging.info(csm.sm_rlzs)

    rlzs = get_csm_rlzs(csm)
    branch_sources = {k: v["sources"] for k, v in rlzs.items()}
    branch_weights = {k: v["weight"] for k, v in rlzs.items()}

    if (branch is not None) and (branch != "iterate"):
        ssm_lt_sources = {branch: branch_sources[branch]}
        ssm_lt_weights = {branch: 1.0}
        ssm_lt_rup_counts = {
            branch: [s.num_ruptures for s in branch_sources[branch]]
        }

    elif branch == "iterate":
        raise NotImplementedError()

    else:
        n_total_sources = sum(
            len(br_source) for br_source in branch_sources.values()
        )
        logging.info(f"Model has {n_total_sources:_} sources")
        sources_w_weights = make_composite_source(
            branch_sources, branch_weights
        )
        ssm_lt_sources = {"composite": list(sources_w_weights.keys())}
        ssm_lt_rup_counts = {
            "composite": [s.num_ruptures for s in ssm_lt_sources["composite"]]
        }
        source_weights = list(sources_w_weights.values())
        ssm_lt_weights = {"composite": []}

        for i, rup_count in enumerate(ssm_lt_rup_counts["composite"]):
            ssm_lt_weights["composite"].append(
                np.ones(rup_count) * source_weights[i]
            )

        ssm_lt_weights["composite"] = np.hstack(ssm_lt_weights["composite"])
        logging.info(
            f"{len(ssm_lt_weights['composite']):_} rups in composite model"
        )

    return ssm_lt_sources, ssm_lt_weights, ssm_lt_rup_counts


def make_composite_source(branch_sources, branch_weights):
    sources_w_weights = AccumDict()
    for br, br_sources in branch_sources.items():
        brr = {src: branch_weights[br] for src in br_sources}
        sources_w_weights += brr

    return sources_w_weights


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
    job_ini_params = {
        "general": {
            "calculation_mode": "preclassical",
            "description": description,
        },
        "calculation": {
            "rupture_mesh_spacing": 2.0,
            "area_source_discretization": 15.0,
            "width_of_mfd_bin": 0.1,  # typically smaller than from cfg; use cfg?
            "maximum_distance": 200,
            "investigation_time": 1.0,
            "source_model_logic_tree": ssm_lt_path,
            "gsim_logic_tree": gmm_lt_path,
        },
        "site_params": {
            "reference_vs30_type": "measured",
            "reference_vs30_value": 800.0,
            "reference_depth_to_1pt0km_per_sec": 30.0,
        },
    }


    job_ini_params_flat = {k:v for k, v in job_ini_params['general'].items()}
    job_ini_params_flat.update(job_ini_params['calculation'])
    job_ini_params_flat.update(job_ini_params['site_params'])

    job_ini_params_flat = {k:str(v) for k, v in job_ini_params_flat.items()}
    job_ini_params_flat['inputs'] = {'source_model_logic_tree': str(ssm_lt_path)}

    return job_ini_params_flat
