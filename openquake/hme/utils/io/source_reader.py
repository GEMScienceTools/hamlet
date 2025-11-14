import os
import logging

from typing import Optional, Sequence

import numpy as np
from openquake.baselib.general import AccumDict
from openquake.calculators.base import run_calc

from openquake.commonlib import datastore
from openquake.commonlib.readinput import get_params
from openquake.engine.engine import create_jobs, run_jobs

from openquake.hme.utils.utils import _get_class_name


from openquake.hme.utils.utils import breakpoint

def csm_from_job_ini(job_ini):
    if not isinstance(job_ini, dict) and os.path.isfile(job_ini):
        job_ini = get_params(job_ini)
        if not job_ini["inputs"].get("site_model", None):
            job_ini["ground_motion_fields"] = False
            job_ini["inputs"]["job_ini"] = "<in-memory>"

    [job] = create_jobs([job_ini])
    job.params["calculation_mode"] = "preclassical"
    run_jobs([job])
    with job, datastore.read(job.calc_id) as dstore:
        csm = dstore["_csm"]
        sources = csm.get_sources()

    return csm, sources, dstore


#def get_sources_by_branch(csm):
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
        rlz_ids = [get_smr_from_trt_smr(trt_smr)
                   for trt_smr in source.trt_smrs]
        for rid in rlz_ids:
            if rid not in rlz_sources:
                rlz_sources[rid] = [source]
            else:
                rlz_sources[rid].append(source)
    return rlz_sources



def get_dstore_rlzs(dstore, csm):
    csm_rlz_groups = {}
    
    # shortcut for 1 rlz
    if len(dstore['full_lt'].sm_rlzs) == 1:
        csm_rlz_groups[0] = {
                'weight': 1.0,
                'sources': csm.get_sources()
                }
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
    collapse_lt: Optional[bool] = False,
    source_types: Optional[Sequence] = None,
    tectonic_region_types: Optional[Sequence] = None,
    description: Optional[str] = None,
):
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

    csm, _sources, dstore = csm_from_job_ini(job_ini)


    rlz_info = {r.ordinal: {'path': r.pid, 'weight': r.weight}
                for r in dstore["full_lt"].sm_rlzs }
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

    if (branch is not None) and (branch != "iterate"): # specific branch
        ssm_lt_sources = {branch: branch_sources[branch]}
        logging.info(
            f"working on branch {branch}, " + 
             f"original weight {branch_weights[branch]}")
        ssm_lt_weights = {branch: 1.0}
        ssm_lt_rup_counts = {
            branch: [s.num_ruptures for s in branch_sources[branch]]
        }

    elif branch == "iterate": # iterate over branches
        raise NotImplementedError()

    else: # no branches specified, i.e. all branches collapsed
        if collapse_lt: # may not work quite right
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
        else:
            ssm_lt_sources = branch_sources
            ssm_lt_rup_counts = {
                    br: [s.num_ruptures for s in srcs]
                    for br, srcs in branch_sources.items()
                    }
            #ssm_lt_weights = {br: [] for br in branch_sources.keys()}
            #for br, br_weight in branch_weights.items():
            #    for num_rups in ssm_lt_rup_counts[br]:
            #        ssm_lt_weights[br].append(np.ones(num_rups) * br_weight)
            ssm_lt_weights = branch_weights

    #breakpoint()

    return ssm_lt_sources, ssm_lt_weights, ssm_lt_rup_counts


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
    #gmm_lt_path = os.path.join(base_dir, gmm_lt_file)
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
