import os
import logging

from typing import Optional, Sequence

import numpy as np
from openquake.baselib.general import AccumDict
from openquake.calculators.base import run_calc

from openquake.commonlib import datastore
from openquake.engine.engine import create_jobs, run_jobs

from openquake.hme.utils.utils import _get_class_name


def csm_from_job_ini(job_ini):
    rups = []
    [job] = create_jobs([job_ini])
    job.params["calculation_mode"] = "preclassical"
    run_jobs([job])
    with job, datastore.read(job.calc_id) as dstore:
        csm = dstore["_csm"]
        sources = csm.get_sources()
        # breakpoint()
        # for src in sources:
        #    logging.info("processing %s", src)
        #    for rup in src.iter_ruptures():
        #        rups.append(rup)

    return csm, sources, dstore


def csm_from_job_ini_old(job_ini):
    if isinstance(job_ini, dict):
        calc = run_calc(
            job_ini,
            # calclation_mode="preclassical",
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
    try:
        grp = csm.get_groups(rlz)
        for g in grp:
            srcs.extend(g)
    except AttributeError:
        srcs.extend(csm.get_sources(rlz))
    return srcs


def get_rlz_source_dstore(rlz, dstore):

    srcs = []
    srcs.extend()


def get_csm_rlzs(csm):
    csm_rlz_groups = {}
    for i, rlz in enumerate(csm.sm_rlzs):
        csm_rlz_groups[i] = {
            "weight": rlz.weight,
            "sources": get_rlz_source(i, csm),
        }
        return csm_rlz_groups


def get_dstore_rlzs(dstore, csm):
    csm_rlz_groups = {}
    for i, rlz in enumerate(dstore["full_lt"].sm_rlzs):
        csm_rlz_groups[i] = {
            "weight": rlz.weight,
            "sources": get_rlz_source(i, csm),
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
        logging.info("Job ini found")
        job_ini = os.path.join(base_dir, job_ini_file)
    else:
        logging.warning("making job ini")
        job_ini = make_job_ini(base_dir, lt_file, gmm_lt_file, description)

    csm, _sources, dstore = csm_from_job_ini(job_ini)

    logging.info("Realizations:")
    logging.info(dstore["full_lt"].sm_rlzs)

    rlzs = get_dstore_rlzs(dstore, csm)
    branch_sources = {k: v["sources"] for k, v in rlzs.items()}
    if source_types is not None:
        logging.info("Filtering sources by type")
        branch_sources = {
            k: filter_sources_by_type(v, source_types)
            for k, v in branch_sources.items()
        }
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


def make_job_ini(
    base_dir: str,
    lt_file: str = "ssmLT.xml",
    gmm_lt_file: str = "gmmLT.xml",
    description: Optional[str] = None,
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

    return job_ini_params_flat
