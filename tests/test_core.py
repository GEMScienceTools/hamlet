import os
import unittest

from openquake.hme.core.core import (read_yaml_config,
                                     get_test_lists_from_config)

BASE_PATH = os.path.dirname(__file__)
UNIT_TEST_DATA_DIR = os.path.join(BASE_PATH, 'data', 'unit_test_data')

yml_2 = os.path.join(UNIT_TEST_DATA_DIR, 'test_core_1.yml')

cfg_2 = read_yaml_config(yml_2)


def test_get_test_lists_from_config():
    test_dict = get_test_lists_from_config(cfg_2)
    gem_test_names = [f.__name__ for f in test_dict['gem']]
    sanity_test_names = [f.__name__ for f in test_dict['sanity']]

    assert gem_test_names == ['mfd_likelihood_test']
    assert sanity_test_names == ['max_check']
