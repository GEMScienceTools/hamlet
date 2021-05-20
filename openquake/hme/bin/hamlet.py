import sys
import logging
import argparse

from openquake.hme.__version__ import __version__

from openquake.hme.core.core import run_tests, read_yaml_config

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main(arg=None):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Hamlet: Hazard Model Evaluation and Testing"
    )

    parser.add_argument("yaml_file")
    parser.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + __version__
    )

    args = parser.parse_args()

    if arg is None:
        yaml_file = args.yaml_file
    cfg = read_yaml_config(yaml_file)
    run_tests(cfg)


if __name__ == "__main__":
    main()
