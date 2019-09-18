import logging

from hztest.utils.bins import SpacemagBin


def _get_mfd_max_mag(mfd: dict) -> float:
    try:
        max_mag = max({mag: rate
                       for mag, rate in mfd.items() if rate > 0.}.keys())
    except ValueError:
        max_mag = 0.
    return max_mag


def _get_max_rupture_mag(sbin: SpacemagBin) -> float:
    rmfd = sbin.get_rupture_mfd()

    return _get_mfd_max_mag(rmfd)


def _get_max_obs_eq_mag(sbin: SpacemagBin) -> float:
    obs_eqs = sbin.get_empirical_mfd()

    return _get_mfd_max_mag(obs_eqs)


def check_bin_max(sbin: SpacemagBin) -> bool:

    obs_eq_max = _get_max_obs_eq_mag(sbin)
    rupture_max = _get_max_rupture_mag(sbin)

    return obs_eq_max < rupture_max
