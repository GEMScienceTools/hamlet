import os
import unittest
from copy import deepcopy

import numpy as np

from openquake.hme.utils import deep_update
from openquake.hme.utils.validate_inputs import validate_cfg
from openquake.hme.core.core import (
    load_obs_eq_catalog,
    load_ruptures_from_ssm,
    load_inputs,
    cfg_defaults,
)

BASE_PATH = os.path.dirname(__file__)

SM1_PATH = os.path.join(BASE_PATH, "data", "source_models", "sm1")
DATA_FILE = os.path.join(SM1_PATH, "data", "phl_synth_catalog.csv")
RUP_CSV = os.path.join(SM1_PATH, "sm1_rups.csv")

# Doing this here because it takes several seconds and should be done once
test_cfg = {
    "meta": {"description": "test"},
    "config": {
        "model_framework": {
            "relm": {
                "N_test": {
                    "prob_model": "poisson",
                    "conf_interval": 0.96,
                    "investigation_time": 40.0,
                },
                "S_test": {
                    "investigation_time": 40.0,
                    "n_iters": 5,
                    "critical_pct": 0.25,
                    "append": True,
                },
            }
        },
        "parallel": False,
        "rand_seed": 69,
    },
    "input": {
        "bins": {
            "h3_res": 3,
            "mfd_bin_min": 6.1,
            "mfd_bin_max": 8.5,
            "mfd_bin_width": 0.2,
        },
        "subset": {
            "file": None,
            "buffer": 0.0,
        },
        "ssm": {
            "ssm_dir": SM1_PATH + "/",
            "ssm_lt_file": "ssmLT.xml",
            #"branch": "b1",  does not work with oq source processing
            "tectonic_region_types": ["Active Shallow Crust"],
            "source_types": None,
        },
        "seis_catalog": {
            "seis_catalog_file": DATA_FILE,
            "stop_date": "2013-01-01",
            "duration": 40.,
            "columns": {
                "time": "time",
                #"source": "Agency",
                "event_id": "eventID",
            },
        },
        "rupture_file": {
            "rupture_file_path": RUP_CSV,
            "read_rupture_file": False,
            "save_rupture_file": False,
        },
    },
}

cfg = deepcopy(cfg_defaults)
cfg = deep_update(cfg, test_cfg)
validate_cfg(cfg)

input_data = load_inputs(cfg)
eq_gdf = load_obs_eq_catalog(cfg)  # repeating but works for some testing
rup_gdf = load_ruptures_from_ssm(cfg)
