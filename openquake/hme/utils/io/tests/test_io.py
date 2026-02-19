import os
import pathlib
import unittest


import numpy as np
import pandas as pd

from openquake.hme.core.core import read_yaml_config
from openquake.hme.utils.io import read_rupture_file
from openquake.hme.utils.simple_rupture import SimpleRupture

from openquake.hme.utils.tests.load_sm1 import cfg, input_data, eq_gdf, rup_gdf

from openquake.hme.utils.io.source_reader import (
    process_source_logic_tree_oq,
)

from openquake.hme.utils.io.source_processing import (
    rupture_dict_from_logic_tree_dict,
    rupture_dict_to_gdf,
)

from openquake.hme.utils.utils import breakpoint

BASE_PATH = pathlib.Path(os.path.dirname(__file__))


def test_read_rupture_file():
    rup_fp = cfg["input"]["rupture_file"]["rupture_file_path"]
    rup_gdf_in = read_rupture_file(rup_fp)
    assert rup_gdf_in.shape == rup_gdf.shape
    n_rows, n_cols = rup_gdf.shape
    for nr in range(n_rows):
        for col in rup_gdf.columns:
            # linux and macos sometimes have 180 deg different strikes
            # for ruptures with near 90 deg dips
            if col != 'strike':
                param_r1 = rup_gdf_in.iloc[nr][col]
                param_r2 = rup_gdf.iloc[nr][col]
                if isinstance(param_r1, str):
                    assert param_r1 == param_r2
                else:
                    np.testing.assert_almost_equal(
                        param_r1, param_r2, decimal=2
                    )


def test_2_branches():
    test_dir = (
        BASE_PATH
        / '..'
        / '..'
        / 'tests'
        / 'data'
        / 'source_models'
        / '2_branches'
    )
    cfg = read_yaml_config(test_dir / 'test_2_ssm_branches.yaml')
    source_cfg = cfg['input']['ssm']
    (
        b_ssm_lt_sources,
        b_ssm_lt_weights,
        b_ssm_lt_rup_counts,
        b_gsim_lt,
    ) = process_source_logic_tree_oq(
        source_cfg["job_ini_file"],
        test_dir / source_cfg["ssm_dir"],
    )

    (
        c_ssm_lt_sources,
        c_ssm_lt_weights,
        c_ssm_lt_rup_counts,
        c_gsim_lt,
    ) = process_source_logic_tree_oq(
        source_cfg["job_ini_file"],
        test_dir / source_cfg["ssm_dir"],
        collapse_lt=True,
    )

    branch_rdf = rupture_dict_to_gdf(
        rupture_dict_from_logic_tree_dict(
            b_ssm_lt_sources,
            b_ssm_lt_rup_counts,
        ),
        b_ssm_lt_weights,
    )

    collapse_rdf = rupture_dict_to_gdf(
        rupture_dict_from_logic_tree_dict(
            c_ssm_lt_sources,
            c_ssm_lt_rup_counts,
        ),
        c_ssm_lt_weights,
    )

    breakpoint()
