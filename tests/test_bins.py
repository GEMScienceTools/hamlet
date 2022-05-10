import os
import logging
import unittest
from copy import deepcopy

import numpy as np

from openquake.hme.utils import deep_update
from openquake.hme.core.core import load_inputs, cfg_defaults
from openquake.hme.model_test_frameworks.relm.relm_tests import (
    S_test,
    N_test,
    M_test,
)

BASE_PATH = os.path.dirname(__file__)
SM1_PATH = os.path.join(BASE_PATH, "data", "source_models", "sm1")
DATA_FILE = os.path.join(SM1_PATH, "data", "phl_eqs.csv")

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
                    "n_iters": 100,
                    "critical_pct": 0.25,
                    "append": True,
                },
            }
        },
        "parallel": False,
        "rand_seed": 69,
    },
    "input": {
        "simple_ruptures": True,
        "bins": {
            "mfd_bin_min": 6.5,
            "mfd_bin_max": 8.5,
            "mfd_bin_width": 0.2,
            "h3_res": 3,
        },
        "subset": {"file": None},
        "ssm": {
            "ssm_dir": SM1_PATH + "/",
            "ssm_lt_file": "ssmLT.xml",
            "branch": "b1",
            "tectonic_region_types": ["Active Shallow Crust"],
            "source_types": None,
        },
        "seis_catalog": {
            "seis_catalog_file": DATA_FILE,
            "columns": {
                "time": ["year", "month", "day", "hour", "minute", "second"],
                "source": "Agency",
                "event_id": "eventID",
            },
        },
        "rupture_file": {
            "rupture_file_path": None,
            "read_rupture_file": False,
            "save_rupture_file": False,
        },
    },
}

cfg = deepcopy(cfg_defaults)
cfg = deep_update(cfg, test_cfg)

bin_gdf, obs_seis_catalog = load_inputs(cfg)


def test_sample_ruptures():
    """
    Test used during debugging, not a good unit test
    """
    sb = bin_gdf.iloc[0].SpacemagBin
    evs = []
    rnd = []
    for i in range(5):
        print(i)
        rs = sb.sample_ruptures(5000.0, clean=True, return_rups=True)
        for k, v in rs.items():
            print(k, len(v))
        evs.append(rs.copy())
        rnd.append(np.random.get_state())
