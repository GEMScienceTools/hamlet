from typing import Optional

from shapely.geometry import Polygon

import numpy as np

from . import utils


class MagBin:
    def __init__(
        self,
        bin_min=None,
        bin_max=None,
        bin_center=None,
        bin_width=None,
        rate=None,
        ruptures=None,
    ):

        self.bin_min = bin_min
        self.bin_max = bin_max
        self.bin_center = bin_center
        self.bin_width = bin_width
        self.rate = rate

        if ruptures is None:
            self.ruptures = []

        self.observed_earthquakes = []
        self.stochastic_earthquakes = []
        self.prospective_earthquakes = []

    def calculate_observed_earthquake_rate(self, t_yrs=1.0, return_rate=False):
        self.observed_earthquake_rate = len(self.observed_earthquakes) / t_yrs
        if return_rate is True:
            return self.observed_earthquake_rate

    def calculate_prospective_earthquake_rate(self, t_yrs=1.0, return_rate=False):
        self.prospective_earthquake_rate = len(self.prospective_earthquakes) / t_yrs
        if return_rate is True:
            return self.prospective_earthquake_rate

    def calculate_total_rupture_rate(self, t_yrs=1, return_rate=False):
        self.net_rupture_rate = sum([r.occurrence_rate for r in self.ruptures]) * t_yrs
        if return_rate is True:
            return self.net_rupture_rate

    def sample_ruptures_old(self, interval_length, t0=0.0, clean=True, 
                        rand_seed=None):
        eqs = utils.flatten_list(
            [
                utils.sample_earthquakes(rup, interval_length, t0, rand_seed)
                for rup in self.ruptures
            ]
        )

        if clean is True:
            self.stochastic_earthquakes = eqs
        else:
            self.stochastic_earthquakes.append(eqs)

    def sample_ruptures(self, interval_length, t0=0.0, clean=True, 
                        rand_seed=None):
        
        eqs = utils.flatten_list(utils.sample_earthquakes_from_ruptures(self.ruptures,
            interval_length, get_event_times=False, rand_seed=rand_seed))

        if clean is True:
            self.stochastic_earthquakes = eqs
        else:
            self.stochastic_earthquakes.append(eqs)

    def to_dict(self):
        params = {
            "bin_min": self.bin_min,
            "bin_max": self.bin_max,
            "bin_center": self.bin_center,
            "rate": self.rate,
        }

        if self.ruptures == []:
            ruptures = {"ruptures": []}
        else:
            ruptures = {"ruptures": [r.__dict__() for r in self.ruptures]}

        if self.observed_earthquakes == []:
            observed_earthquakes = {"observed_earthquakes": []}
        else:
            observed_earthquakes = {
                "observed_earthquakes": [e.__dict__ for e in self.observed_earthquakes]
            }

        if self.prospective_earthquakes == []:
            prospective_earthquakes = {"prospective_earthquakes": []}
        else:
            prospective_earthquakes = {
                "prospective_earthquakes": [
                    e.__dict__ for e in self.prospective_earthquakes
                ]
            }

        return {**params, **ruptures, **observed_earthquakes, **prospective_earthquakes}


class SpacemagBin:
    def __init__(
        self,
        poly: Polygon,
        min_mag=None,
        max_mag=None,
        bin_width=0.2,
        bin_id=None,
        mag_bin_centers=None,
    ):

        self.poly = poly
        self.bin_id = bin_id
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.bin_width = bin_width
        self.mag_bin_centers = mag_bin_centers

        self.make_mag_bins()
        self.stochastic_earthquakes = {bc: [] for bc in self.mag_bin_centers}
        self.observed_earthquakes = {bc: [] for bc in self.mag_bin_centers}
        self.prospective_earthquakes = {bc: [] for bc in self.mag_bin_centers}

    def make_mag_bins(self):
        if self.mag_bin_centers is None:
            self.mag_bin_centers = [self.min_mag]
            bc = self.min_mag
            while bc <= self.max_mag:
                bc += self.bin_width
                self.mag_bin_centers.append(np.round(bc, 2))

        self.mag_bins = {
            bc: MagBin(
                bin_center=bc,
                bin_width=self.bin_width,
                bin_min=bc - self.bin_width / 2,
                bin_max=bc + self.bin_width / 2,
            )
            for bc in self.mag_bin_centers
        }

    def get_bin_edges(self):
        edges = [mb.bin_min for mb in self.mag_bins.values()]

        edges.append(self.mag_bin_centers[-1] + self.bin_width / 2.0)

        return edges

    def to_dict(self):
        params = {
            "poly": {
                "x": self.poly.exterior.coords.xy[0].tolist(),
                "y": self.poly.exterior.coords.xy[1].tolist(),
            },
            "min_mag": self.min_mag,
            "max_mag": self.max_mag,
            "bin_width": self.bin_width,
            "bin_id": self.bin_id,
            "mag_bin_centers": self.mag_bin_centers,
        }

        mag_bins = {
            "mag_bins": {bc: mb.to_dict() for (bc, mb) in self.mag_bins.items()}
        }

        return {**params, **mag_bins}

    def sample_ruptures(
        self,
        interval_length: float,
        t0: float = 0.0,
        clean: bool = True,
        return_rups: bool = False,
        rand_seed=None
    ):
        for bc, mag_bin in self.mag_bins.items():
            mag_bin.sample_ruptures(interval_length, t0=t0, clean=clean,
                                    rand_seed=rand_seed)
            if clean is True:
                self.stochastic_earthquakes[bc] = mag_bin.stochastic_earthquakes
            else:
                self.stochastic_earthquakes[bc].append(mag_bin.stochastic_earthquakes)

        if return_rups is True:
            return self.stochastic_earthquakes
        else:
            return

    def get_rupture_mfd(self, cumulative=False):

        # may not be returned in order in Python < 3.5
        noncum_mfd = {
            bc: self.mag_bins[bc].calculate_total_rupture_rate(return_rate=True)
            for bc in self.mag_bin_centers
        }

        cum_mfd = {}
        cum_mag = 0.0
        # dict has descending order
        for cb in self.mag_bin_centers[::-1]:
            cum_mag += noncum_mfd[cb]
            cum_mfd[cb] = cum_mag

        # make new dict with ascending order
        cum_mfd = {cb: cum_mfd[cb] for cb in self.mag_bin_centers}

        self.cum_mfd = cum_mfd
        self.noncum_mfd = noncum_mfd

        if cumulative is False:
            return self.noncum_mfd
        else:
            return self.cum_mfd

    def get_rupture_sample_mfd(
        self, interval_length, t0=0.0, normalize=True, cumulative=False,
        rand_seed=None
    ):

        self.sample_ruptures(interval_length=interval_length, t0=t0,
                             rand_seed=rand_seed)

        if normalize is True:
            denom = interval_length
        else:
            denom = 1
        noncum_mfd = {
            bc: len(eqs) / denom for bc, eqs in self.stochastic_earthquakes.items()
        }

        cum_mfd = {}
        cum_mag = 0.0
        # dict has descending order
        for cb in self.mag_bin_centers[::-1]:
            cum_mag += noncum_mfd[cb]
            cum_mfd[cb] = cum_mag

        # make new dict with ascending order
        cum_mfd = {cb: cum_mfd[cb] for cb in self.mag_bin_centers}

        self.stochastic_noncum_mfd = noncum_mfd
        self.stochastic_cum_mfd = cum_mfd

        if cumulative is False:
            return self.stochastic_noncum_mfd
        else:
            return self.stochastic_cum_mfd

    def get_empirical_mfd(self, t_yrs=1.0, cumulative=False):
        """
        Calculates the MFD of empirical (observed) earthquakes; no fitting.
        """

        # may not be returned in order in Python < 3.5
        noncum_mfd = {
            bc: self.mag_bins[bc].calculate_observed_earthquake_rate(
                t_yrs=t_yrs, return_rate=True
            )
            for bc in self.mag_bin_centers
        }

        cum_mfd = {}
        cum_mag = 0.0
        # dict has descending order
        for cb in self.mag_bin_centers[::-1]:
            cum_mag += noncum_mfd[cb]
            cum_mfd[cb] = cum_mag

        # make new dict with ascending order
        cum_mfd = {cb: cum_mfd[cb] for cb in self.mag_bin_centers}

        self.obs_cum_mfd = cum_mfd
        self.obs_noncum_mfd = noncum_mfd

        if cumulative is False:
            return self.obs_noncum_mfd
        else:
            return self.obs_cum_mfd

    def get_prospective_mfd(self, t_yrs=1.0, cumulative=False):
        """
        Calculates the MFD of prospective earthquakes; no fitting.
        """

        # may not be returned in order in Python < 3.5
        noncum_mfd = {
            bc: self.mag_bins[bc].calculate_prospective_earthquake_rate(
                t_yrs=t_yrs, return_rate=True
            )
            for bc in self.mag_bin_centers
        }

        cum_mfd = {}
        cum_mag = 0.0
        # dict has descending order
        for cb in self.mag_bin_centers[::-1]:
            cum_mag += noncum_mfd[cb]
            cum_mfd[cb] = cum_mag

        # make new dict with ascending order
        cum_mfd = {cb: cum_mfd[cb] for cb in self.mag_bin_centers}

        self.pro_cum_mfd = cum_mfd
        self.pro_noncum_mfd = noncum_mfd

        if cumulative is False:
            return self.pro_noncum_mfd
        else:
            return self.pro_cum_mfd
