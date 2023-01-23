import unittest

import numpy as np
import pandas as pd

from openquake.hme.utils.io import read_rupture_file
from openquake.hme.utils.simple_rupture import SimpleRupture

from openquake.hme.utils.tests.load_sm1 import cfg, input_data, eq_gdf, rup_gdf


def test_read_rupture_file():
    rup_fp = cfg["input"]["rupture_file"]["rupture_file_path"]
    rup_gdf_in = read_rupture_file(rup_fp)
    assert rup_gdf_in.shape == rup_gdf.shape
    n_rows, n_cols = rup_gdf.shape
    for nr in range(n_rows):
        for col in rup_gdf.columns:
            param_r1 = rup_gdf_in.iloc[nr][col]
            param_r2 = rup_gdf.iloc[nr][col]
            if isinstance(param_r1, str):
                assert param_r1 == param_r2
            else:
                np.testing.assert_almost_equal(param_r1, param_r2)
