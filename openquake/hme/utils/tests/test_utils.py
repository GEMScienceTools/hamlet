import os
import datetime
import unittest

import numpy as np

from openquake.hme.utils import (
    get_mag_duration_from_comp_table,
    flatten_list,
    rupture_list_to_gdf,
    SimpleRupture,
    make_earthquake_gdf_from_csv,
    get_model_mfd,
    get_obs_mfd,
)

BASE_PATH = os.path.dirname(__file__)
test_data_dir = os.path.join(BASE_PATH, "data", "source_models", "sm1")


class TestBasicUtils(unittest.TestCase):
    def test_flatten_list(self):
        lol = [["l"], ["o"], ["l"]]
        flol = flatten_list(lol)
        self.assertEqual(flol, ["l", "o", "l"])

    def test_get_mag_duration_from_comp_table(self):
        comp_table = [
            [2010.0, 5.0],
            [1980.0, 5.5],
            [1970.0, 6.0],
            [1960.0, 6.5],
            [1950.0, 7.0],
            [1900.0, 8.0],
        ]

        end_year = datetime.date(2023, 1, 1)

        assert (
            round(
                get_mag_duration_from_comp_table(comp_table, 6.7, end_year), 1
            )
            == 63.0
        )

        assert (
            round(
                get_mag_duration_from_comp_table(comp_table, 8.7, end_year), 1
            )
            == 123.0
        )
