import logging
from typing import Optional

from geopandas import GeoDataFrame, GeoSeries

from .sanity_test_functions import (
    max_check_function,
)


def min_max_check():
    raise NotImplementedError


def max_check(cfg, input_data, framework="sanity"):
    bin_width = cfg["input"]["bins"]["mfd_bin_width"]

    test_cfg = cfg["config"]["model_framework"][framework]
    warn = test_cfg.get("warn", True)
    parallel = cfg["config"].get("parallel", True)

    return max_check_function(
        input_data, bin_width, warn=warn, parallel=parallel
    )


sanity_test_dict = {
    "max_check": max_check,
}
