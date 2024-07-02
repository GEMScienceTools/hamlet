import pathlib
import dateutil
import datetime


DAYS_PER_YEAR = 365.2425


def validate_cfg(cfg: dict) -> None:

    check_fix_seis_catalog(cfg["input"]["seis_catalog"])


def check_fix_seis_catalog(seis_cat_cfg) -> None:

    if not seis_cat_cfg.get("completeness_table"):

        if "start_date" in seis_cat_cfg:
            seis_cat_cfg["start_date"] = check_fix_date(
                seis_cat_cfg["start_date"]
            )

        if "stop_date" in seis_cat_cfg:
            seis_cat_cfg["stop_date"] = check_fix_date(
                seis_cat_cfg["stop_date"]
            )

        if "duration" in seis_cat_cfg:
            if "start_date" in seis_cat_cfg and "stop_date" in seis_cat_cfg:
                check_duration(
                    seis_cat_cfg["start_date"],
                    seis_cat_cfg["stop_date"],
                    seis_cat_cfg["duration"],
                )
        else:
            seis_cat_cfg["duration"] = (
                seis_cat_cfg["stop_date"] - seis_cat_cfg["start_date"]
            )


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
