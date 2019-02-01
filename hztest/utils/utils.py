import numpy as np
import pandas as pd
import geopandas as gpd

from multiprocessing import Pool
import os

from shapely.geometry import Point

from .stats import sample_events_in_interval



def _flatten_list(lol):
    """
    Flattens a list of lists (lol)
    """
    return [item for sublist in lol for item in sublist]


def rupture_dict_from_logic_dict(ld, source_types=('simple_fault')):
    return {br: rupture_list_from_lt_branch(branch, source_types)
            for br, branch in ld.items()}
        

def rupture_list_from_lt_branch(branch, source_types=('simple_fault')):
    # iterates over the logic dict (one per branch of logic tree),
    # gets all of the ruptures, adds their source as an attribute,

    rupture_list = []
    
    def process_rup(rup, source):
        rup.source = source.source_id
        return rup
        
    for source_type, sources in branch.items():
        if source_type in source_types and sources != []:
            rups = [process_rup(r, source) for source in sources
                                           for r in source.iter_ruptures()]

            rupture_list.extend(rups)

    return rupture_list

def _process_rup(rup, source):
    rup.source = source.source_id
    return rup

def _process_source(source):
    return [_process_rup(r, source) for r in source.iter_ruptures()]

def rupture_list_from_lt_branch_parallel(branch, source_types=('simple_fault')):
    # iterates over the logic dict (one per branch of logic tree),
    # gets all of the ruptures, adds their source as an attribute,

    rupture_list = []
    
    for source_type, sources in branch.items():
        if source_type in source_types and sources != []:
            with Pool(os.cpu_count()-1) as pool:
                rups = pool.map(_process_source, sources)
                rups = _flatten_list(rups)
                rupture_list.extend(rups)

    return rupture_list


def rupture_list_to_gdf(rupture_list):
    
    df = pd.DataFrame(index=range(len(rupture_list)),
                      data=rupture_list, columns=['rupture'])

    df['geometry'] = df.apply(lambda z: Point(z.rupture.hypocenter.longitude, 
                                              z.rupture.hypocenter.latitude),
                              axis=1)
    return gpd.GeoDataFrame(df)


def make_spatial_bins_df_from_file(bin_fp):
    """
    Returns a geopandas dataframe from a file containing spatial bins as
    polygons.
    """

    bin_df = gpd.read_file(bin_fp)

    return bin_df


def add_ruptures_to_bins(rupture_gdf, bin_df):

    join_df = gpd.sjoin(rupture_gdf, bin_df, how='left')

    rupture_gdf['bin_id'] = join_df['index_right']

    def bin_row(row):
        if not np.isnan(row.bin_id): 
            spacemag_bin = bin_df.loc[row.bin_id, 'SpacemagBin']
            nearest_bc = _nearest_bin(row.rupture.mag, spacemag_bin.mag_bin_centers)
            spacemag_bin.mag_bins[nearest_bc].ruptures.append(row.rupture)

    _ = rupture_gdf.apply(bin_row, axis=1)


def make_earthquake_gdf(earthquake_df):
    pass


def _nearest_bin(val, bin_centers):
    bca = np.array(bin_centers)

    return bin_centers[np.argmin(np.abs(val-bca))]


def add_earthquakes_to_bins(earthquake_gdf, bin_df):
    # observed_earthquakes, not ruptures

    join_df = gpd.sjoin(earthquake_gdf, bin_df, how='left')

    earthquake_gdf['bin_id'] = join_df['index_right']

    for i, eq in earthquake_gdf.iterrows():
        if not np.isnan(eq.bin_id): 
            spacemag_bin = bin_df.loc[eq.bin_id, 'SpacemagBin']
            nearest_bc = _nearest_bin(eq.Eq.mag, spacemag_bin.mag_bin_centers)

            spacemag_bin.mag_bins[nearest_bc].observed_earthquakes.append(
                                                                    eq['Eq'])
            spacemag_bin.observed_earthquakes[nearest_bc].append(eq['Eq'])


def make_SpacemagBins_from_bin_df(bin_df, min_mag=6., max_mag=9.,
                                  bin_width=0.1,):
    def bin_to_mag(row):
        return SpacemagBin(row.geometry, bin_id=row._name, min_mag=min_mag,
                            max_mag=max_mag)
    bin_df['SpacemagBin'] = bin_df.apply(bin_to_mag, axis=1)


class Earthquake():
    def __init__(self, mag=None, latitude=None, longitude=None, depth=None,
                 time=None, source=None, event_id=None):
        self.mag = mag
        self.latitude = latitude
        self.longitude = longitude
        self.depth = depth
        self.time = time
        self.source = source
        self.event_id = event_id


def make_earthquakes(rupture, interval_length, t0=0.):
    event_times = sample_events_in_interval(rupture.occurrence_rate,
                                            interval_length, t0)
    try:
        source = rupture.source
    except:
        source = None
    
    eqs = [Earthquake(mag=rupture.mag, latitude=rupture.hypocenter.latitude,
                      longitude=rupture.hypocenter.longitude, 
                      depth=rupture.hypocenter.depth,
                      source=source, time=et)
                      for et in event_times]
    return eqs


class MagBin():
    def __init__(self, bin_min=None, bin_max=None, bin_center=None, 
                 bin_width=None, rate=None, ruptures=None):

        self.bin_min = bin_min
        self.bin_max = bin_max
        self.rate = rate

        if ruptures is None:
            self.ruptures = []

        self.observed_earthquakes = []
        self.stochastic_earthquakes = []

    def calculate_observed_earthquake_rate(self, t_yrs=1., return_rate=False):
        self.observed_earthquake_rate = len(self.observed_earthquakes) / t_yrs
        if return_rate is True:
            return self.observed_earthquake_rate

    def calculate_total_rupture_rate(self, t_yrs=1, return_rate=False):
        self.net_rupture_rate = sum([r.occurrence_rate 
                                     for r in self.ruptures]) * t_yrs
        if return_rate is True:
            return self.net_rupture_rate
                

    def sample_ruptures(self, interval_length, t0=0., clean=True):
        eqs = _flatten_list([make_earthquakes(rup, interval_length, t0)
                             for rup in self.ruptures])
        
        if clean is True:
            self.stochastic_earthquakes = eqs
        else:
            self.stochastic_earthquakes.append(eqs)


class SpacemagBin():
    def __init__(self, poly, min_mag=None, max_mag=None, bin_width=0.1, 
                 bin_id=None, mag_bin_centers=None):

        self.poly = poly
        self.bin_id = bin_id
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.bin_width = bin_width
        self.mag_bin_centers = mag_bin_centers

        self.make_mag_bins()
        self.observed_earthquakes = {bc: [] for bc in self.mag_bin_centers}
        self.stochastic_earthquakes = {bc: [] for bc in self.mag_bin_centers}

    def make_mag_bins(self):
        if self.mag_bin_centers is None:
            self.mag_bin_centers = [self.min_mag]
            bc = self.min_mag
            while bc <= self.max_mag:
                bc += self.bin_width
                self.mag_bin_centers.append(bc)

        self.mag_bins = {bc: MagBin(bin_center=bc, bin_width=self.bin_width,
                                    bin_min=bc-self.bin_width/2, 
                                    bin_max=bc+self.bin_width/2)
                         for bc in self.mag_bin_centers}

    def sample_ruptures(self, interval_length, t0=0., clean=True):
        for bc, mag_bin in self.mag_bins.items():
            mag_bin.sample_ruptures(interval_length, t0=t0, clean=clean)
            if clean is True:
                self.stochastic_earthquakes[bc] = mag_bin.stochastic_earthquakes
            else:
                self.stochastic_earthquakes[bc].append(
                                                 mag_bin.stochastic_earthquakes)

    def get_rupture_mfd(self, cumulative=False):

        # may not be returned in order in Python < 3.5
        noncum_mfd = {bc: self.mag_bins[bc].calculate_total_rupture_rate(
                                                                    return_rate=True)
                      for bc in self.mag_bin_centers}

        cum_mfd = {}
        cum_mag = 0.
        # dict has descending order
        for cb in self.mag_bin_centers[::-1]:
            cum_mag += noncum_mfd[cb]
            cum_mfd[cb] = cum_mag

        # make new dict with ascending order
        cum_mfd = {cb: cum_mfd[cb] for cb in self.mag_bin_centers}
 
        self.cum_mfd = cum_mfd
        self.noncum_mfd = noncum_mfd
 
        if cumulative is False:
            return noncum_mfd
        else:
            return cum_mfd


    def get_empirical_mfd(self, t_yrs=1., cumulative=False):
        """
        Calculates the MFD of empirical (observed) earthquakes; no fitting.
        """

        # may not be returned in order in Python < 3.5
        noncum_mfd = {bc: 
                self.mag_bins[bc].calculate_observed_earthquake_rate(t_yrs=t_yrs,
                                                                     return_rate=True)
                      for bc in self.mag_bin_centers}

        cum_mfd = {}
        cum_mag = 0.
        # dict has descending order
        for cb in self.mag_bin_centers[::-1]:
            cum_mag += noncum_mfd[cb]
            cum_mfd[cb] = cum_mag

        # make new dict with ascending order
        cum_mfd = {cb: cum_mfd[cb] for cb in self.mag_bin_centers}
 
        self.cum_mfd = cum_mfd
        self.noncum_mfd = noncum_mfd
 
        if cumulative is False:
            return noncum_mfd
        else:
            return cum_mfd


