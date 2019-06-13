import sys
import logging

from hztest.core.core import run_tests, read_yaml_config

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    cfg = read_yaml_config(sys.argv[1])
    run_tests(cfg)