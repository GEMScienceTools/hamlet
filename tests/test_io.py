import unittest

import numpy as np
import pandas as pd

from openquake.hme.utils.io import read_rupture_file
from openquake.hme.utils.simple_rupture import SimpleRupture

from .load_sm1 import cfg, bin_gdf, eq_gdf, rup_gdf


#@unittest.skip("need to handle relative import")
def test_read_rupture_file():
    rup_fp = cfg["input"]["rupture_file"]["rupture_file_path"]
    rup_gdf_in = read_rupture_file(rup_fp)
    #pd.testing.assert_frame_equal(rup_gdf, rup_gdf_in)
    assert rup_gdf_in.shape == rup_gdf.shape
    n_rows, n_cols = rup_gdf.shape
    for nr in range(n_rows):
        for nc in range(n_cols):
            r1 = rup_gdf_in.iloc[nr, nc]
            r2 = rup_gdf.iloc[nr, nc]
            np.testing.assert_almost_equal(r1.strike, r2.strike)
            np.testing.assert_almost_equal(r1.dip, r2.dip)
            np.testing.assert_almost_equal(r1.rake, r2.rake)
            np.testing.assert_almost_equal(r1.mag, r2.mag)
            np.testing.assert_almost_equal(r1.hypocenter.x, r2.hypocenter.x)
            np.testing.assert_almost_equal(r1.hypocenter.y, r2.hypocenter.y)
            np.testing.assert_almost_equal(r1.hypocenter.z, r2.hypocenter.z)
            np.testing.assert_almost_equal(r1.occurrence_rate,
                                           r2.occurrence_rate)
            assert r1.source == r2.source
