import sys
import logging
import argparse
import warnings
import traceback

warnings.simplefilter(action="ignore", category=FutureWarning)

from openquake.hme.__version__ import __version__

from openquake.hme.core.core import run_tests, read_yaml_config
from openquake.hme.utils.log import init_logging, check_for_log, add_logfile

try:
    import ipdb

    debugger = ipdb
except ImportError:
    import pdb

    debugger = pdb

root_logger = init_logging()


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
    parser.add_argument(
        "--pdb",
        action="store_true",
        help="Drop into pdb post-mortem on exception",
    )

    args = parser.parse_args()

    if arg is None:
        yaml_file = args.yaml_file

    try:
        # first without validation for logging
        cfg = read_yaml_config(yaml_file, validate=False)

        logfile = check_for_log(cfg)
        if logfile:
            add_logfile(logfile, root_logger)
            logger.info(f"logging to {logfile}")

        # now for real
        cfg = read_yaml_config(yaml_file, validate=True)
        run_tests(cfg)
    except Exception:
        logger.error("An error occurred during execution")
        traceback.print_exc()

        if args.pdb:
            debugger.post_mortem()


if __name__ == "__main__":
    main()
