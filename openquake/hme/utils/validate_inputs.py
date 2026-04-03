import os
import pathlib
import dateutil
import datetime
import logging


DAYS_PER_YEAR = 365.2425


def validate_cfg(cfg: dict) -> None:
    check_fix_seis_catalog(cfg["input"]["seis_catalog"])
    convert_deprecated_parameters(cfg)
    check_branch_config(cfg)
    check_flatfile_requires_gmm_lt(cfg)


def check_branch_config(cfg: dict) -> None:
    branch = cfg["input"]["ssm"].get("branch")
    if branch == "iterate":
        logging.info(
            "Branch iteration mode enabled: "
            "will evaluate each branch independently"
        )


def check_flatfile_requires_gmm_lt(cfg: dict) -> None:
    if "flatfile" not in cfg["input"]:
        return

    # Get the ssm key of the config
    ssm_cfg = cfg["input"]["ssm"]

    # If specifying a job file instead of SSC + SSC LT
    # then gsim_lt will be in there so return
    if ssm_cfg.get("job_ini_file") is not None:
        return

    # Otherwise need to make sure a gmmLT is in the ssm_dir
    gmm_lt_path = os.path.join(ssm_cfg["ssm_dir"], "gmmLT.xml")
    if not os.path.isfile(gmm_lt_path):
        raise FileNotFoundError(
            f"A flatfile is specified but no GMM logic tree file found at "
            f"'{gmm_lt_path}'. A gmmLT.xml file must be present in the "
            f"ssm_dir when using ground motion evaluation."
        )


def check_fix_seis_catalog(seis_cat_cfg) -> None:
    if not seis_cat_cfg.get("completeness_table"):
        get_date_parameters(seis_cat_cfg)


def convert_deprecated_parameters(cfg: dict) -> None:
    """
    Convert deprecated parameter names to their current equivalents.
    Currently handles:
    - 'critical_pct' to 'critical_frac'
    - 'percentile' to 'fractile'

    This allows backward compatibility with old configuration files.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary
    """
    # Only process if model_framework exists
    if "config" not in cfg or "model_framework" not in cfg["config"]:
        return

    frameworks = cfg["config"]["model_framework"]

    for framework_name, framework in frameworks.items():
        # Skip if not a dictionary
        if not isinstance(framework, dict):
            continue

        for test_name, test_config in framework.items():
            # Skip if not a dictionary
            if not isinstance(test_config, dict):
                continue

            if "critical_pct" in test_config:
                test_config["critical_frac"] = test_config.pop("critical_pct")
                logging.warning(
                    f"Deprecated parameter 'critical_pct' in {framework_name}.{test_name} "
                    f"has been converted to 'critical_frac'. Please update your configuration."
                )


def get_date_parameters(seis_cat_cfg) -> None:
    """
    Ensures that for any combination of start_date, stop_date and duration,
    the missing parameter is added to the seis_cat_cfg.

    Parameters
    ----------
    seis_cat_cfg : dict
        Configuration dictionary for seismic catalog
    """
    if "start_date" in seis_cat_cfg:
        seis_cat_cfg["start_date"] = check_fix_date(seis_cat_cfg["start_date"])

    if "stop_date" in seis_cat_cfg:
        seis_cat_cfg["stop_date"] = check_fix_date(seis_cat_cfg["stop_date"])

    if "duration" in seis_cat_cfg:
        if "start_date" in seis_cat_cfg and "stop_date" in seis_cat_cfg:
            check_duration(
                seis_cat_cfg["start_date"],
                seis_cat_cfg["stop_date"],
                seis_cat_cfg["duration"],
            )
        elif "start_date" in seis_cat_cfg:
            # Calculate stop_date from start_date and duration
            delta_days = int(seis_cat_cfg["duration"] * DAYS_PER_YEAR)
            seis_cat_cfg["stop_date"] = seis_cat_cfg[
                "start_date"
            ] + datetime.timedelta(days=delta_days)
        elif "stop_date" in seis_cat_cfg:
            # Calculate start_date from stop_date and duration
            delta_days = int(seis_cat_cfg["duration"] * DAYS_PER_YEAR)
            seis_cat_cfg["start_date"] = seis_cat_cfg[
                "stop_date"
            ] - datetime.timedelta(days=delta_days)
    else:
        if "start_date" in seis_cat_cfg and "stop_date" in seis_cat_cfg:
            seis_cat_cfg["duration"] = (
                seis_cat_cfg["stop_date"] - seis_cat_cfg["start_date"]
            ).days / DAYS_PER_YEAR


def check_seis_catalog_path(seis_cat_cfg) -> None:
    if not pathlib.Path(seis_cat_cfg["seis_catalog_file"]).exists():
        raise Exception


def check_fix_date(date):
    if isinstance(date, (datetime.datetime, datetime.date)):
        pass

    elif isinstance(date, int):
        try:
            date_str = "{}-1-1".format(date)
            date = dateutil.parser.parse(date_str)
        except:
            err_msg = "cannot convert {} to date".format(date)
            raise ValueError(err_msg)
    elif isinstance(date, str):
        try:
            date = dateutil.parser.parse(date)
        except:
            err_msg = "cannot convert {} to date".format(date)
            raise ValueError(err_msg)

    else:
        err_msg = "cannot convert {} to date".format(date)
        raise ValueError(err_msg)

    return date


def check_duration(start_date, stop_date, duration):
    years_diff = (stop_date - start_date).days / DAYS_PER_YEAR

    if abs(years_diff - duration) > 0.5:
        err_msg = (
            "Seis catalog duration {} does not match start "
            "and stop dates.  Please fix or remove one piece "
            "of information".format(duration)
        )
        raise ValueError(err_msg)
