"""
Functionality for writing reports demonstrating the results of the Hamlet
testing.
"""

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from jinja2 import Environment, FileSystemLoader

from openquake.hme.utils.plots import plot_likelihood_map

BASE_DATA_PATH = os.path.dirname(__file__)
template_dir = os.path.join(BASE_DATA_PATH, 'templates')


def _init_env() -> Environment:
    file_loader = FileSystemLoader(template_dir)
    env = Environment(loader=file_loader)
    return env


def generate_basic_report(cfg: dict,
                          results: dict,
                          bin_gdf: Optional[GeoDataFrame] = None,
                          eq_gdf: Optional[GeoDataFrame] = None) -> None:
    """
    Generates an HTML report with the results of the various evaluations or
    tests performed by Hamlet.

    :param cfg:
        Configuration from parsed yaml file.

    :type cfg:
        `dict`

    """

    env = _init_env()
    report_template = env.get_template('basic_report.html')

    render_result_text(env=env,
                       cfg=cfg,
                       results=results,
                       bin_gdf=bin_gdf,
                       eq_gdf=eq_gdf)

    report = report_template.render(cfg=cfg, results=results)

    with open(cfg['report']['basic']['outfile'], 'w') as report_file:
        report_file.write(report)


def render_result_text(env: Environment,
                       cfg: dict,
                       results: dict,
                       bin_gdf: Optional[GeoDataFrame] = None,
                       eq_gdf: Optional[GeoDataFrame] = None) -> None:

    if 'model_mfd' in results.keys():
        render_mfd(env=env, cfg=cfg, results=results)

    if 'likelihood' in results.keys():
        render_likelihood(env=env,
                          cfg=cfg,
                          results=results,
                          bin_gdf=bin_gdf,
                          eq_gdf=eq_gdf)

    if 'max_mag_check' in results.keys():
        render_max_mag(env=env, cfg=cfg, results=results)


def render_mfd(env: Environment, cfg: dict, results: dict):
    mfd_template = env.get_template('mfd.html')
    results['model_mfd']['rendered_text'] = mfd_template.render(
        cfg=cfg, results=results)


def render_likelihood(env: Environment,
                      cfg: dict,
                      results: dict,
                      bin_gdf: GeoDataFrame,
                      eq_gdf: Optional[GeoDataFrame] = None) -> None:

    total_log_like = np.sum(bin_gdf['log_like']) / bin_gdf.shape[0]
    total_log_like = "{0:2f}".format(total_log_like)

    if 'plot_eqs' in cfg['report']['basic'].keys():
        plot_eqs = cfg['report']['basic']['plot_eqs']
    else:
        plot_eqs = True

    likelihood_map_str = plot_likelihood_map(bin_gdf, plot_eqs, eq_gdf)

    likelihood_template = env.get_template('likelihood.html')

    results['likelihood']['rendered_text'] = likelihood_template.render(
        cfg=cfg,
        results=results,
        total_log_like=total_log_like,
        likelihood_map_str=likelihood_map_str)


def render_max_mag(env: Environment, cfg: dict, results: dict) -> None:

    if results['max_mag_check']['val'] == []:
        max_mag_results = (
            'PASS: All bins produce seismicity greater than observed.')
    else:
        max_mag_results = (
            "Bins {} have higher observed seismicity than they can produce.".
            format(results['max_mag_check']['val']))

    max_mag_template = env.get_template('max_mag_check.html')
    results['max_mag_check']['rendered_text'] = max_mag_template.render(
        max_mag_results=max_mag_results)
