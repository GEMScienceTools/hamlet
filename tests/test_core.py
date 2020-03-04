import os
import unittest

from openquake.hme.core.core import (read_yaml_config,
                                     get_test_list_from_config,
                                     get_test_lists_from_config)

yml_1 = './model_test_frameworks/gem/integration_tests/gem_sm1/test_sm1_poisson.yml'
yml_2 = './data/unit_test_data/test_core_1.yml'

cfg_1 = read_yaml_config(yml_1)
cfg_2 = read_yaml_config(yml_2)

tests_1 = get_test_list_from_config(cfg_1)


def test_get_test_lists_from_config():
    test_dict = get_test_lists_from_config(cfg_2)
    gem_test_names = [f.__name__ for f in test_dict['gem']]
    sanity_test_names = [f.__name__ for f in test_dict['sanity']]

    assert gem_test_names == ['mfd_likelihood_test']
    assert sanity_test_names == ['max_check']
