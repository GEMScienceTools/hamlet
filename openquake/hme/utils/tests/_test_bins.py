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

from openquake.hme.utils.tests.load_sm1 import cfg, input_data, eq_gdf, rup_gdf

