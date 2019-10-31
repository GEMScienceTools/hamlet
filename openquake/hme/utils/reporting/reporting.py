import os
from typing import Optional

from jinja2 import Environment, FileSystemLoader
from geopandas import GeoDataFrame

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
    env = _init_env()
    report_template = env.get_template('basic_report.html')

    render_result_text(env=env, cfg=cfg, results=results)

    report = report_template.render(cfg=cfg, results=results)

    with open(cfg['report']['basic']['outfile'], 'w') as report_file:
        report_file.write(report)


def render_result_text(env: Environment,
                       cfg: dict,
                       results: dict,
                       bin_gdf: Optional[GeoDataFrame] = None,
                       eq_gdf: Optional[GeoDataFrame] = None):

    if 'model_mfd' in results.keys():
        render_mfd(env=env, cfg=cfg, results=results)

    if 'likelihood' in results.keys():
        render_likelihood(env=env,
                          cfg=cfg,
                          results=results,
                          bin_gdf=bin_gdf,
                          eq_gdf=eq_gdf)


def render_mfd(env: Environment, cfg: dict, results: dict):
    mfd_template = env.get_template('mfd.html')
    results['model_mfd']['rendered_text'] = mfd_template.render(
        cfg=cfg, results=results)


def render_likelihood(env: Environment,
                      cfg: dict,
                      results: dict,
                      bin_gdf: GeoDataFrame,
                      eq_gdf: Optional[GeoDataFrame] = None):

    likelihood_template = env.get_template('likelihood.html')
    results['likelihood']['rendered_text'] = likelihood_template.render(
        cfg=cfg, results=results)
