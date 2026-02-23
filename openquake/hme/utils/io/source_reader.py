import os
import logging

from typing import Optional, Sequence

import numpy as np
from openquake.baselib.general import AccumDict

from openquake.commonlib import datastore
from openquake.commonlib.readinput import (
        get_params,
        get_composite_source_model,
        )
from openquake.engine.engine import create_jobs, run_jobs

from openquake.hazardlib.gsim_lt import GsimLogicTree
from openquake.hazardlib.source import MultiPointSource

from openquake.hme.utils.utils import _get_class_name, breakpoint

try:
    from openquake.hazardlib.source_group import read_csm

    csm_new_flag = True
except ImportError:
    csm_new_flag = False


def csm_from_job_ini(job_ini, get_gsim_lt: bool = False):
    if not isinstance(job_ini, dict) and os.path.isfile(job_ini):
        job_ini = get_params(job_ini)
        if not job_ini["inputs"].get("site_model", None):
            job_ini["ground_motion_fields"] = False
            job_ini["inputs"]["job_ini"] = "<in-memory>"

    logging.debug(job_ini)

    logging.debug("creating job")
    [job] = create_jobs([job_ini])
    logging.debug("\tcreated job")
    logging.debug(job)
    logging.debug("setting calculation mode")
    job.params["calculation_mode"] = "preclassical"
    logging.debug("\tset calculation mode")
    logging.debug("running job")
    run_jobs([job])
    logging.debug("\tran job")
    logging.debug("getting csm from dstore")
    with job, datastore.read(job.calc_id) as dstore:
        if csm_new_flag:
            csm = read_csm(dstore)
        else:  # older OQ
            csm = dstore['_csm']
        sources = csm.get_sources()
        logging.debug("\tgot csm from dstore")

        if get_gsim_lt:
            gmm_lt_filepath = job.params["inputs"]["gsim_logic_tree"]
        else:
            gmm_lt_filepath = None

    return csm, sources, dstore, gmm_lt_filepath


# def get_sources_by_branch(csm):
#    bs = {'null': []}
#    for src in csm.get_sources():
#        try:
#            src_branch = src.branch
#            if src_branch not in bs:
#                bs[src_branch] = []
#            bs[src_branch].append(src)
#        except AttributeError:
#            bs['null'].append(src)
#    return bs


def get_smr_from_trt_smr(trt_smr):
    # calculate the mask to isolate the source_model_rlz_index
    mask = (1 << 24) - 1
    # Extract the source_model_rlz_index by applying the mask
    source_model_rlz_index = trt_smr & mask
    return source_model_rlz_index


def get_sources_by_rlz_idx(sources):
    # would be nice to modify this to use the rlz path not id
    rlz_sources = {}
    for source in sources:
        rlz_ids = [
            get_smr_from_trt_smr(trt_smr) for trt_smr in source.trt_smrs
        ]
        for rid in rlz_ids:
            if rid not in rlz_sources:
                rlz_sources[rid] = [source]
            else:
                rlz_sources[rid].append(source)
    return rlz_sources


def get_dstore_rlzs(dstore, csm):
    csm_rlz_groups = {}

    # shortcut for 1 rlz
    if len(dstore["full_lt"].sm_rlzs) == 1:
        csm_rlz_groups[0] = {"weight": 1.0, "sources": csm.get_sources()}
        return csm_rlz_groups

    srcs_by_rlz = get_sources_by_rlz_idx(csm.get_sources())
    #    breakpoint()

    for i, rlz in enumerate(dstore["full_lt"].sm_rlzs):

        csm_rlz_groups[i] = {
            "weight": rlz.weight,
            "sources": srcs_by_rlz[i],
        }

    return csm_rlz_groups


def filter_sources_by_type(sources, source_types):
    if source_types is None:
        return sources

    filtered_sources = []
    for src in sources:
        if _get_class_name(src) in source_types:
            filtered_sources.append(src)

    return filtered_sources


rupfields = dict(
    mag=np.float32,
    occurrence_rate=np.float32,
    hypo_lon=np.float32,
    hypo_lat=np.float32,
    hypo_dep=np.float32,
    grp_id=np.uint16,
    rup_id=np.uint32,
    src_id=np.uint32,
    # add more if you like
)


def process_source_logic_tree_oq(
    job_ini_file,
    base_dir: str,
    lt_file: str = "ssmLT.xml",
    gmm_lt_file: str = "gmmLT.xml",
    sites_file: Optional[str] = None,
    branch: Optional[str] = None,
    collapse_lt: Optional[bool] = True,
    source_types: Optional[Sequence] = None,
    tectonic_region_types: Optional[Sequence] = None,
    description: Optional[str] = None,
    get_gsim_lt: bool = False,
):
    logging.debug("we are at the beginning of process_source_logic_tree_oq")
    if job_ini_file is not None:
        logging.info("Job ini found")
        job_ini = os.path.join(base_dir, job_ini_file)
    else:
        logging.warning("making job ini")
        job_ini = make_job_ini(
            base_dir,
            lt_file=lt_file,
            gmm_lt_file=gmm_lt_file,
            description=description,
            sites_file=sites_file,
        )

    csm, _sources, dstore, gmm_lt_filepath = csm_from_job_ini(
        job_ini, get_gsim_lt=get_gsim_lt
    )

    rlz_info = {
        r.ordinal: {"path": r.pid, "weight": r.weight}
        for r in dstore["full_lt"].sm_rlzs
    }
    logging.info("Realizations:")
    logging.info(rlz_info)

    rlzs = get_dstore_rlzs(dstore, csm)
    branch_sources = {k: v["sources"] for k, v in rlzs.items()}
    if source_types is not None:
        logging.info("Filtering sources by type")
        branch_sources = {
            k: filter_sources_by_type(v, source_types)
            for k, v in branch_sources.items()
        }
    branch_weights = {k: v["weight"] for k, v in rlzs.items()}

    if (branch is not None) and (branch != "iterate"):  # specific branch
        ssm_lt_sources = {branch: branch_sources[branch]}
        logging.info(
            f"working on branch {branch}, "
            + f"original weight {branch_weights[branch]}"
        )
        ssm_lt_weights = {branch: 1.0}
        ssm_lt_rup_counts = {
            branch: [s.num_ruptures for s in branch_sources[branch]]
        }

    elif branch == "iterate":
        raise ValueError(
            "branch='iterate' should be handled by run_tests_iterate, "
            "not called directly through process_source_logic_tree_oq"
        )

    else:  # no branches specified, i.e. all branches collapsed
        if collapse_lt:  # may not work quite right
            n_total_sources = sum(
                len(br_source) for br_source in branch_sources.values()
            )
            logging.info(f"Model has {n_total_sources:_} sources")
            sources_w_weights = make_composite_source(
                branch_sources, branch_weights
            )
            ssm_lt_sources = {"composite": list(sources_w_weights.keys())}
            ssm_lt_rup_counts = {
                "composite": [
                    s.num_ruptures for s in ssm_lt_sources["composite"]
                ]
            }
            src_weight_dict = {}
            for src, w in sources_w_weights.items():
                if isinstance(src, MultiPointSource):
                    for sub_src in src:
                        src_weight_dict[sub_src.source_id] = w
                else:
                    src_weight_dict[src.source_id] = w

            ssm_lt_weights = {"composite": src_weight_dict}
            logging.info(
                f"{len(ssm_lt_weights['composite']):_} sources in composite model"
            )
        else:
            ssm_lt_sources = branch_sources
            ssm_lt_rup_counts = {
                br: [s.num_ruptures for s in srcs]
                for br, srcs in branch_sources.items()
            }
            # ssm_lt_weights = {br: [] for br in branch_sources.keys()}
            # for br, br_weight in branch_weights.items():
            #    for num_rups in ssm_lt_rup_counts[br]:
            #        ssm_lt_weights[br].append(np.ones(num_rups) * br_weight)
            ssm_lt_weights = branch_weights

    # breakpoint()

    if get_gsim_lt:
        gsim_lt = read_gsim_lt(gmm_lt_filepath)
    else:
        gsim_lt = None

    return ssm_lt_sources, ssm_lt_weights, ssm_lt_rup_counts, gsim_lt


def prepare_iterate_branches(
    job_ini_file,
    base_dir: str,
    lt_file: str = "ssmLT.xml",
    gmm_lt_file: str = "gmmLT.xml",
    sites_file: Optional[str] = None,
    source_types: Optional[Sequence] = None,
    tectonic_region_types: Optional[Sequence] = None,
    description: Optional[str] = None,
    get_gsim_lt: bool = False,
):
    """
    Read the CSM once and return per-branch source info for iterate mode.

    This avoids re-running the expensive preclassical OQ calculation for
    each branch when iterating over logic tree branches.

    Returns a tuple of (branch_sources, branch_rup_counts, rlz_info, gsim_lt)
    where branch_sources and branch_rup_counts are dicts keyed by branch
    ordinal, and rlz_info contains the original weights and paths.
    """
    logging.info("Preparing iterate branches (reading CSM once)")

    if job_ini_file is not None:
        job_ini = os.path.join(base_dir, job_ini_file)
    else:
        job_ini = make_job_ini(
            base_dir,
            lt_file=lt_file,
            gmm_lt_file=gmm_lt_file,
            description=description,
            sites_file=sites_file,
        )

    csm, _sources, dstore, gmm_lt_filepath = csm_from_job_ini(
        job_ini, get_gsim_lt=get_gsim_lt
    )

    rlz_info = {
        r.ordinal: {"path": r.pid, "weight": r.weight}
        for r in dstore["full_lt"].sm_rlzs
    }
    logging.info(f"Found {len(rlz_info)} realizations: {rlz_info}")

    rlzs = get_dstore_rlzs(dstore, csm)
    branch_sources = {k: v["sources"] for k, v in rlzs.items()}

    if source_types is not None:
        logging.info("Filtering sources by type")
        branch_sources = {
            k: filter_sources_by_type(v, source_types)
            for k, v in branch_sources.items()
        }

    branch_rup_counts = {
        br: [s.num_ruptures for s in srcs]
        for br, srcs in branch_sources.items()
    }

    if get_gsim_lt:
        gsim_lt = read_gsim_lt(gmm_lt_filepath)
    else:
        gsim_lt = None

    return branch_sources, branch_rup_counts, rlz_info, gsim_lt


def make_composite_source(branch_sources, branch_weights):
    sources_w_weights = AccumDict()
    for br, br_sources in branch_sources.items():
        brr = {src: branch_weights[br] for src in br_sources}
        sources_w_weights += brr

    return sources_w_weights


def make_job_ini(
    base_dir: str,
    lt_file: str = "ssmLT.xml",
    gmm_lt_file: str = "gmmLT.xml",
    description: Optional[str] = None,
    sites_file: Optional[str] = None,
):
    ssm_lt_path = os.path.join(base_dir, lt_file)
    # gmm_lt_path = os.path.join(base_dir, gmm_lt_file)
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
            # "gsim_logic_tree": gmm_lt_path,
            "ground_motion_fields": False,
            "truncation_level": 3.0,
            "intensity_measure_types_and_levels": {"PGA": [0.5]},
            "ground_motion_fields": False,
        },
        "site_params": {
            "reference_vs30_type": "measured",
            "reference_vs30_value": 800.0,
            "reference_depth_to_1pt0km_per_sec": 30.0,
        },
    }

    job_ini_params_flat = {k: v for k, v in job_ini_params["general"].items()}
    job_ini_params_flat.update(job_ini_params["calculation"])
    job_ini_params_flat.update(job_ini_params["site_params"])

    job_ini_params_flat = {k: str(v) for k, v in job_ini_params_flat.items()}
    job_ini_params_flat["inputs"] = {
        "job_ini": "<in-memory>",
        "source_model_logic_tree": str(ssm_lt_path),
    }

    if sites_file:
        job_ini_params_flat["inputs"] = ["sites_file"]

    return job_ini_params_flat


def read_gsim_lt(gsim_filepath, tectonic_region_types=["*"], ltnode=None):

    logging.info("Reading gsim_lt")
    gsim_lt = GsimLogicTree(
        gsim_filepath,
        tectonic_region_types=tectonic_region_types,
        ltnode=ltnode,
    )

    return gsim_lt
