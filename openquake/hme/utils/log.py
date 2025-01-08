import os
import logging

from logging.handlers import RotatingFileHandler

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = logging.INFO


def init_logging():
    logging.basicConfig(
        format=LOG_FORMAT,
        level=LOG_LEVEL,
        datefmt=DATE_FORMAT,
    )

    # Get the root logger
    return logging.getLogger()


def check_for_log(cfg):
    try:
        logfile = cfg["config"]["log_file"]
    except KeyError:
        logfile = None
    return logfile


def add_logfile(logfile, logger):
    try:
        if os.path.exists(logfile):
            os.remove(logfile)
    except OSError as e:
        logger.warn(f"Error removing old log file: {e}")

    file_handler = logging.FileHandler(logfile)

    file_handler.setFormatter(
        logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=DATE_FORMAT,
        )
    )

    # Add handler to root logger
    logging.getLogger().addHandler(file_handler)
