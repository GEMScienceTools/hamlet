import sys
import logging

from openquake.hme.core.core import run_tests, read_yaml_config

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S')


def main(arg=None):
    if arg is None:
        yaml_file = sys.argv[1]
    cfg = read_yaml_config(yaml_file)
    run_tests(cfg)


if __name__ == '__main__':
    main()
