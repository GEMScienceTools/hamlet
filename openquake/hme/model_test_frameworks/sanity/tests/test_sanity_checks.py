import unittest
from copy import deepcopy

import numpy as np
from shapely.geometry import Polygon
from openquake.hazardlib.source import ParametricProbabilisticRupture as Rupture
from openquake.hazardlib.geo import Point

# from openquake.hme.utils import SpacemagBin, Earthquake
from openquake.hme.model_test_frameworks.sanity.sanity_checks import (
    min_max_check,
    max_check,
)

# from openquake.hme.model_test_frameworks.sanity.sanity_test_functions import (
# check_bin_max,
# _get_mfd_max_mag,
# )


def SpacemagBin():
    return


def Earthquake():
    return


@unittest.skip("deprecated function; need to replace")
def test_check_bin_max():
    bin_1 = SpacemagBin(Polygon(), min_mag=6.0, max_mag=7.0, bin_width=0.2)

    bin_1.mag_bins[6.0].ruptures.append(
        Rupture(6.0, "undefined", None, Point(0.0, 0.0), None, 0.01, None)
    )

    bin_1.mag_bins[6.2].ruptures.append(
        Rupture(6.2, "undefined", None, Point(0.0, 0.0), None, 0.005, None)
    )

    bin_1.mag_bins[6.0].observed_earthquakes.append(Earthquake())

    bin_2 = deepcopy(bin_1)

    bin_2.mag_bins[7.0].observed_earthquakes.append(Earthquake())

    assert check_bin_max(bin_1) == True
    assert check_bin_max(bin_2) == False


@unittest.skip("deprecated function; need to replace")
def test_get_mfd_max_mag():
    d_rates = {6.0: 0.1, 6.2: 0.05, 6.4: 0.0}
    d_no_rates = {6.0: 0.0, 6.2: 0.0, 6.4: 0.0}

    assert _get_mfd_max_mag(d_rates) == 6.2
    assert _get_mfd_max_mag(d_no_rates) == 0.0
