"""
Utility functions for running tests in the RELM model test framework.
"""
from typing import Sequence, Dict, List

import numpy as np
from geopandas import GeoSeries

from openquake.hme.utils import SpacemagBin, parallelize
