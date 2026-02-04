"""
Tests for completeness table handling in MFD evaluations and statistical tests.

These tests verify that:
1. MFD evaluations correctly handle completeness tables with annualization
2. N test properly accounts for varying observation periods per magnitude bin
3. M test properly accounts for varying observation periods per magnitude bin
4. Plotting functions correctly handle per-bin durations
"""

import unittest
import datetime
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame

from openquake.hme.utils import (
    get_model_mfd,
    get_obs_mfd,
    get_mag_duration_from_comp_table,
)
from openquake.hme.model_test_frameworks.gem.gem_test_functions import (
    model_mfd_eval_fn,
)
from openquake.hme.model_test_frameworks.relm.relm_test_functions import (
    n_test_function,
    m_test_function,
)
from openquake.hme.utils.plots import _make_stoch_mfds


class TestCompletenessTableHandling(unittest.TestCase):
    """Test completeness table handling across different components."""

    def setUp(self):
        """Set up test data with a simple rupture set and completeness table."""
        # Create simple rupture data
        self.rupture_data = pd.DataFrame(
            {
                "magnitude": [5.0, 5.5, 6.0, 6.5, 7.0],
                "occurrence_rate": [1.0, 0.5, 0.1, 0.05, 0.01],  # annual rates
                "longitude": [0.0, 0.0, 0.0, 0.0, 0.0],
                "latitude": [0.0, 0.0, 0.0, 0.0, 0.0],
                "depth": [10.0, 10.0, 10.0, 10.0, 10.0],
            }
        )
        self.rup_gdf = GeoDataFrame(self.rupture_data)

        # Create observed earthquake data
        # M5+: 50 events (should span 50 years)
        # M6+: 5 events (should span 100 years)
        # M7+: 1 event (should span 150 years)
        self.eq_data = pd.DataFrame(
            {
                "magnitude": [5.0] * 45
                + [6.0] * 4
                + [7.0] * 1
                + [5.5] * 5,  # 50 M5+, 5 M6+, 1 M7+
                "time": [datetime.datetime(2000, 1, 1)] * 55,
                "longitude": [0.0] * 55,
                "latitude": [0.0] * 55,
                "depth": [10.0] * 55,
            }
        )
        self.eq_gdf = GeoDataFrame(self.eq_data)

        # Completeness table: [[year, magnitude], ...]
        # M5+ complete from 1975 (50 years to 2025)
        # M6+ complete from 1925 (100 years to 2025)
        # M7+ complete from 1875 (150 years to 2025)
        self.completeness_table = [
            [1975, 5.0],
            [1925, 6.0],
            [1875, 7.0],
        ]

        self.stop_date = datetime.date(2025, 1, 1)

        # Magnitude bins
        self.mag_bins = {
            5.0: (5.0, 5.5),
            5.5: (5.5, 6.0),
            6.0: (6.0, 6.5),
            6.5: (6.5, 7.0),
            7.0: (7.0, 7.5),
        }

    def test_get_mag_duration_from_comp_table(self):
        """Test that duration calculation from completeness table works correctly."""
        # M5.0 should have 50 years
        duration_5 = get_mag_duration_from_comp_table(
            self.completeness_table, 5.0, self.stop_date
        )
        self.assertAlmostEqual(duration_5, 50.0, delta=0.1)

        # M6.0 should have 100 years
        duration_6 = get_mag_duration_from_comp_table(
            self.completeness_table, 6.0, self.stop_date
        )
        self.assertAlmostEqual(duration_6, 100.0, delta=0.1)

        # M7.0 should have 150 years
        duration_7 = get_mag_duration_from_comp_table(
            self.completeness_table, 7.0, self.stop_date
        )
        self.assertAlmostEqual(duration_7, 150.0, delta=0.1)

    def test_model_mfd_with_completeness_table(self):
        """Test that model MFD correctly applies completeness table durations."""
        # Get model MFD with completeness table (non-annualized)
        model_mfd = get_model_mfd(
            self.rup_gdf,
            self.mag_bins,
            completeness_table=self.completeness_table,
            stop_date=self.stop_date,
        )

        # Expected: annual_rate * duration for each bin
        # M5.0: 1.0/yr * 50yr = 50.0
        # M6.0: 0.1/yr * 100yr = 10.0
        # M7.0: 0.01/yr * 150yr = 1.5
        self.assertAlmostEqual(model_mfd[5.0], 50.0, places=1)
        self.assertAlmostEqual(model_mfd[6.0], 10.0, places=1)
        self.assertAlmostEqual(model_mfd[7.0], 1.5, places=1)

    def test_obs_mfd_with_completeness_table_annualized(self):
        """Test that observed MFD correctly annualizes with completeness table."""
        # Get observed MFD with annualization
        obs_mfd = get_obs_mfd(
            self.eq_gdf,
            self.mag_bins,
            completeness_table=self.completeness_table,
            stop_date=self.stop_date,
            annualize=True,
        )

        # Expected: count / duration for each bin
        # M5.0: 45 events / 50 years = 0.9/yr
        # M6.0: 4 events / 100 years = 0.04/yr
        # M7.0: 1 event / 150 years = 0.0067/yr
        self.assertAlmostEqual(obs_mfd[5.0], 0.9, places=1)
        self.assertAlmostEqual(obs_mfd[6.0], 0.04, places=2)
        self.assertAlmostEqual(obs_mfd[7.0], 0.0067, places=3)

    def test_model_mfd_eval_fn_with_annualization(self):
        """Test that model_mfd_eval_fn correctly handles annualization."""
        # Test with annualize=True
        result_annualized = model_mfd_eval_fn(
            self.rup_gdf,
            self.eq_gdf,
            self.mag_bins,
            t_yrs=50.0,
            completeness_table=self.completeness_table,
            stop_date=self.stop_date,
            annualize=True,
        )

        mfd_df = result_annualized["test_data"]["mfd_df"]

        # Model should be annual rates (annualize=True sets t_yrs=1.0)
        # Observed should be annualized (counts / duration)
        self.assertAlmostEqual(mfd_df.loc[5.0, "mod_mfd"], 1.0, places=1)
        self.assertAlmostEqual(mfd_df.loc[6.0, "mod_mfd"], 0.1, places=2)
        self.assertAlmostEqual(mfd_df.loc[7.0, "mod_mfd"], 0.01, places=2)

        self.assertAlmostEqual(mfd_df.loc[5.0, "obs_mfd"], 0.9, places=1)
        self.assertAlmostEqual(mfd_df.loc[6.0, "obs_mfd"], 0.04, places=2)

    def test_model_mfd_eval_fn_without_annualization(self):
        """Test that model_mfd_eval_fn correctly handles non-annualized case."""
        # Test with annualize=False
        result_non_annualized = model_mfd_eval_fn(
            self.rup_gdf,
            self.eq_gdf,
            self.mag_bins,
            t_yrs=50.0,
            completeness_table=self.completeness_table,
            stop_date=self.stop_date,
            annualize=False,
        )

        mfd_df = result_non_annualized["test_data"]["mfd_df"]

        # Model should be expected counts (annual_rate * duration)
        # Observed should be raw counts
        self.assertAlmostEqual(mfd_df.loc[5.0, "mod_mfd"], 50.0, places=1)
        self.assertAlmostEqual(mfd_df.loc[6.0, "mod_mfd"], 10.0, places=1)
        self.assertAlmostEqual(mfd_df.loc[7.0, "mod_mfd"], 1.5, places=1)

        self.assertAlmostEqual(mfd_df.loc[5.0, "obs_mfd"], 45.0, places=0)
        self.assertAlmostEqual(mfd_df.loc[6.0, "obs_mfd"], 4.0, places=0)
        self.assertAlmostEqual(mfd_df.loc[7.0, "obs_mfd"], 1.0, places=0)

    def test_n_test_with_completeness_table(self):
        """Test that N test correctly handles completeness tables."""
        test_config = {
            "mag_bins": self.mag_bins,
            "completeness_table": self.completeness_table,
            "stop_date": self.stop_date,
            "prob_model": "poisson",
            "conf_interval": 0.95,
        }

        result = n_test_function(self.rup_gdf, self.eq_gdf, test_config)

        # Expected counts:
        # Model: sum of (annual_rate * duration) = 50 + 25 + 10 + 7.5 + 1.5 = 94
        # Observed: should be calculated from MFD with completeness table
        # The test should properly account for varying durations

        self.assertIn("n_obs_earthquakes", result)
        self.assertIn("n_pred_earthquakes", result)
        self.assertIn("test_pass", result)

        # Observed should be around 55 (our total event count)
        self.assertAlmostEqual(result["n_obs_earthquakes"], 55, delta=1)

        # Model prediction should account for completeness table durations
        # Should be close to our expected 94
        self.assertGreater(result["n_pred_earthquakes"], 80)
        self.assertLess(result["n_pred_earthquakes"], 100)

    def test_m_test_with_completeness_table(self):
        """Test that M test correctly handles completeness tables."""
        result = m_test_function(
            self.rup_gdf,
            self.eq_gdf,
            self.mag_bins,
            t_yrs=50.0,
            n_iters=100,
            completeness_table=self.completeness_table,
            stop_date=self.stop_date,
            not_modeled_likelihood=1e-5,
            critical_frac=0.25,
            normalize_n_eqs=True,
        )

        # M test should return valid results
        self.assertIn("fractile", result)
        self.assertIn("test_res", result)
        self.assertIn("critical_frac", result)

        # Fractile should be between 0 and 1
        self.assertGreaterEqual(result["fractile"], 0.0)
        self.assertLessEqual(result["fractile"], 1.0)

    def test_stoch_mfd_with_per_bin_durations(self):
        """Test that stochastic MFD generation handles per-bin durations."""
        # Create a simple cumulative MFD
        mfd = {5.0: 1.0, 6.0: 0.1, 7.0: 0.01}

        # Test with uniform duration
        stoch_mfds_uniform = _make_stoch_mfds(mfd, iters=10, t_yrs=50.0)
        self.assertEqual(len(stoch_mfds_uniform), 10)
        self.assertEqual(len(stoch_mfds_uniform[0]), 3)

        # Test with per-bin durations (dict)
        t_yrs_dict = {5.0: 50.0, 6.0: 100.0, 7.0: 150.0}
        stoch_mfds_varying = _make_stoch_mfds(mfd, iters=10, t_yrs=t_yrs_dict)
        self.assertEqual(len(stoch_mfds_varying), 10)
        self.assertEqual(len(stoch_mfds_varying[0]), 3)

        # With longer durations, uncertainty should be smaller (counts higher then divided by longer period)
        # This is a qualitative check - we expect the function to run without error
        self.assertTrue(all(isinstance(val, (int, float)) for smfd in stoch_mfds_varying for val in smfd))


class TestCompletenessTableEdgeCases(unittest.TestCase):
    """Test edge cases for completeness table handling."""

    def test_no_completeness_table(self):
        """Test that functions work correctly without completeness table."""
        rup_data = pd.DataFrame(
            {
                "magnitude": [5.0, 6.0],
                "occurrence_rate": [1.0, 0.1],
                "longitude": [0.0, 0.0],
                "latitude": [0.0, 0.0],
                "depth": [10.0, 10.0],
            }
        )
        rup_gdf = GeoDataFrame(rup_data)

        mag_bins = {5.0: (5.0, 5.5), 6.0: (6.0, 6.5)}

        # Should work with just t_yrs
        model_mfd = get_model_mfd(rup_gdf, mag_bins, t_yrs=50.0)
        self.assertAlmostEqual(model_mfd[5.0], 50.0, places=1)
        self.assertAlmostEqual(model_mfd[6.0], 5.0, places=1)

    def test_annualize_flag_consistency(self):
        """Test that annualize flag is consistently returned in results."""
        rup_data = pd.DataFrame(
            {
                "magnitude": [5.0],
                "occurrence_rate": [1.0],
                "longitude": [0.0],
                "latitude": [0.0],
                "depth": [10.0],
            }
        )
        rup_gdf = GeoDataFrame(rup_data)

        eq_data = pd.DataFrame(
            {
                "magnitude": [5.0] * 10,
                "time": [datetime.datetime(2000, 1, 1)] * 10,
                "longitude": [0.0] * 10,
                "latitude": [0.0] * 10,
                "depth": [10.0] * 10,
            }
        )
        eq_gdf = GeoDataFrame(eq_data)

        mag_bins = {5.0: (5.0, 5.5)}

        # Test annualize=True
        result_true = model_mfd_eval_fn(
            rup_gdf, eq_gdf, mag_bins, t_yrs=10.0, annualize=True
        )
        self.assertTrue(result_true["test_data"]["annualize"])

        # Test annualize=False
        result_false = model_mfd_eval_fn(
            rup_gdf, eq_gdf, mag_bins, t_yrs=10.0, annualize=False
        )
        self.assertFalse(result_false["test_data"]["annualize"])


if __name__ == "__main__":
    unittest.main()
