import os
import logging

import matplotlib
#matplotlib.use('svg')
matplotlib.pyplot.switch_backend('svg')

from hztest.core.core import run_tests, read_yaml_config

logging.basicConfig(level=logging.INFO)

BASE_PATH = os.path.dirname(__file__)
#test_data_dir = os.path.join(BASE_PATH, 'tests', 'model_test_framworks', 'gem',
#                             'integration_tests', 'gem_sm1')

test_file = os.path.join(BASE_PATH, 'test_sm1_empirical.yml')


def test_read_cfg():
    cfg = read_yaml_config(test_file)

    assert True


def test_run_tests():

    cfg = read_yaml_config(test_file)

    curr_dir = os.getcwd()

    os.chdir(BASE_PATH)

    run_tests(cfg)

    os.chdir(curr_dir)

    assert True
