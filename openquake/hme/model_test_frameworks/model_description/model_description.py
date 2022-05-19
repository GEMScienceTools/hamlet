from typing import Mapping, Sequence, Tuple

import numpy as np

from openquake.hme.utils import (
    get_model_mfd,
    get_obs_mfd,
    get_mag_bins,
    get_mag_bins_from_cfg,
)


def _get_inclusive_min_max_mags(
    mags: Sequence[float], cfg: Mapping[str, Mapping[str, float]]
) -> Tuple[float, float]:
    """Finds the minimum and maximum bin centers that fully encompass
    the minimum and maximum magnitudes of the data (i.e. all data are within one
    half-width of the returned bin centers, including the edges.)

    :param mags: Earthquake magnitudes
    :type mags: Sequence[float]
    :param cfg: Model configuration
    :type cfg: Mapping[str, Mapping[str, float]]
    :return: minimum and maximum
    :rtype: Tuple[float]
    """

    mfd_min_mag = round(cfg["input"]["bins"]["mfd_bin_min"], 2)
    mfd_max_mag = round(cfg["input"]["bins"]["mfd_bin_max"], 2)
    mfd_bin_width = cfg["input"]["bins"]["mfd_bin_width"]
    half_width = mfd_bin_width / 2.0

    data_min_mag = np.min(mags)
    data_max_mag = np.max(mags)

    total_min_mag: float = mfd_min_mag
    total_max_mag: float = mfd_min_mag  # ensure intervals line up
    while (total_min_mag - half_width) >= data_min_mag:
        total_min_mag -= mfd_bin_width

    while (total_max_mag + half_width) < max(data_max_mag, mfd_max_mag):
        total_max_mag += mfd_bin_width

    # if total_max_mag > max(data_max_mag, mfd_max_mag):
    #    total_max_mag -= mfd_bin_width

    total_min_mag = round(total_min_mag, 2)
    total_max_mag = round(total_max_mag, 2)

    return total_min_mag, total_max_mag


def describe_mfds(cfg, input_data):

    model_min_mag, model_max_mag = _get_inclusive_min_max_mags(
        input_data["rupture_gdf"].magnitude, cfg
    )
    model_mag_bins = get_mag_bins(
        model_min_mag, model_max_mag, cfg["input"]["bins"]["mfd_bin_width"]
    )

    obs_min_mag, obs_max_mag = _get_inclusive_min_max_mags(
        input_data["eq_gdf"].magnitude, cfg
    )
    obs_mag_bins = get_mag_bins(
        obs_min_mag, obs_max_mag, cfg["input"]["bins"]["mfd_bin_width"]
    )

    # need to scale by investigation time to compare
    model_mfd = get_model_mfd(input_data["rupture_gdf"], model_mag_bins)
    obs_mfd = get_obs_mfd(
        input_data["eq_gdf"],
        obs_mag_bins,
        t_yrs=cfg["input"]["seis_catalog"]["duration"],
    )

    mfds = {"model_mfd": model_mfd, "obs_mfd": obs_mfd}

    return mfds


def describe_model(cfg, input_data):
    model_description = {"model_mfds": describe_mfds(cfg, input_data)}

    return model_description


model_description_test_dict = {
    "describe_mfds": describe_mfds,
    "describe_model": describe_model,
}
