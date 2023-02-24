import numpy as np

from openquake import hme
from openquake.hme.model_test_frameworks import model_description


def test_get_inclusive_min_max_mags():
    test_mags_1 = np.array([5.9, 6.0, 6.8, 7.2])
    test_mags_2 = np.array([5.9, 6.0, 6.8, 7.3])
    test_mags_3 = np.array([5.69999999999, 6.0, 6.8, 7.3])

    cfg_1 = {
        "input": {"mfd_bin_min": 6.0, "mfd_bin_max": 9.1, "mfd_bin_width": 0.2}
    }
    cfg_2 = {
        "input": {"mfd_bin_min": 6.0, "mfd_bin_max": 6.8, "mfd_bin_width": 0.2}
    }

    assert model_description._get_inclusive_min_max_mags(
        test_mags_1, cfg_1
    ) == (
        5.8,
        9.2,
    )

    assert model_description._get_inclusive_min_max_mags(
        test_mags_1, cfg_2
    ) == (
        5.8,
        7.2,
    )

    assert model_description._get_inclusive_min_max_mags(
        test_mags_2, cfg_2
    ) == (
        5.8,
        7.2,
    )

    assert model_description._get_inclusive_min_max_mags(
        test_mags_3, cfg_2
    ) == (
        5.6,
        7.2,
    )
