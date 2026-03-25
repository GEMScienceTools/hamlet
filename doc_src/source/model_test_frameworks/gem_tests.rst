.. _gem-tests-evaluations:


GEM Tests and Evaluations
=========================

These tests are developed by GEM, some based on the literature (e.g. Zechar et
al. 2010), some based on GEM's own ideas and implementations. See
:mod:`~openquake.hme.model_test_frameworks.gem` for the function documentation.
Sanity checks (as detailed below) are also available from the GEM testing
framework, for convenience during the workflow.

The GEM framework includes the N, M, S, and L consistency tests (similar to the
RELM/CSEP versions but with configurable handling of unmodeled cells/bins), as
well as several additional evaluations: MFD comparison, moment rate analysis,
rupture matching, cumulative occurrence, and ground motion evaluation.

.. note::

   The ``likelihood`` test is deprecated and should not be used. Use the
   ``M_test``, ``S_test``, and ``L_test`` instead, which provide more robust
   and well-characterized likelihood-based consistency tests.


Statistical Consistency Tests
-----------------------------

These tests evaluate model consistency using Monte Carlo simulations. Each test
generates many stochastic catalogs from the model and compares a test statistic
of the observed catalog against the distribution of the stochastic catalogs.


.. _gem-N-test:

N-Test
~~~~~~

Compares the total number of observed earthquakes to the number predicted by the
model. The observed count is checked against a confidence interval derived from
the Poisson (or cumulative Poisson) distribution.

Parameters:

``conf_interval``
    Confidence interval for the test (e.g. ``0.95`` means 95%).

``prob_model``
    Probability model: ``"poisson"`` or ``"poisson_cum"``.

``prospective``
    Optional. If ``True``, use the prospective catalog instead of the
    retrospective catalog. Default: ``False``.


.. _gem-M-test:

M-Test
~~~~~~

Evaluates the consistency of the magnitude-frequency distribution of the model
vs. the observations. The log-likelihood of the observed earthquakes given the
model forecast is compared with the log-likelihoods of stochastic catalogs
generated from the same forecast. If the observed log-likelihood falls below the
``critical_frac`` threshold of the stochastic distribution, the test fails.

The log-likelihoods are calculated for each magnitude bin using the Poisson
distribution, then aggregated as the geometric mean across bins.

This test is based on Zechar et al. (2010) with two differences: (1) the total
number of earthquakes in stochastic simulations is not fixed, and (2) the
geometric mean is used instead of the product of bin likelihoods. Neither
difference affects pass/fail outcomes meaningfully.

Parameters:

``critical_frac``
    Fraction of simulations below which the test fails (e.g. ``0.025``
    for a 2.5th percentile threshold).

``n_iters``
    Number of Monte Carlo iterations.

``prospective``
    Optional. Default: ``False``.

``normalize_n_eqs``
    Optional. Normalize the number of earthquakes in stochastic catalogs
    to match the observed count. Default: ``True``.

``not_modeled_likelihood``
    Optional. Likelihood assigned to magnitude bins with zero modeled rate
    but observed earthquakes. Default: ``1e-5``.


.. _gem-S-test:

S-Test
~~~~~~

Evaluates the spatial consistency of the model by comparing per-cell likelihoods
of the observed catalog against stochastic catalogs. This highlights spatial
cells where the model over- or under-predicts seismicity.

Parameters:

``critical_frac``
    Fraction threshold for test failure.

``n_iters``
    Number of Monte Carlo iterations.

``prospective``
    Optional. Default: ``False``.

``normalize_n_eqs``
    Optional. Normalize by number of earthquakes. Default: ``False``.

``not_modeled_likelihood``
    Optional. Default: ``1e-5``.

``likelihood_function``
    Optional. The likelihood function to use for per-cell evaluation. Options:
    ``"mfd"`` (default) or ``"conf_interval_poisson"``.


.. _gem-L-test:

L-Test
~~~~~~

Joint likelihood test combining spatial and magnitude information. This is the
most comprehensive consistency test, evaluating the overall likelihood of the
observed catalog given the model.

Parameters:

``critical_frac``
    Fraction threshold for test failure.

``n_iters``
    Number of Monte Carlo iterations.

``prospective``
    Optional. Default: ``False``.

``not_modeled_likelihood``
    Optional. Default: ``1e-5``.


Magnitude-Frequency Distribution Evaluations
---------------------------------------------

.. _gem-model-mfd-eval:

Model MFD Evaluation (``model_mfd``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sums up the MFDs from all spatial cells to produce a total model MFD, which is
compared to the observed MFD from the earthquake catalog. This produces a figure
in the report showing both MFDs. Use ``{}`` for default parameters.

Parameters (all optional):

``investigation_time``
    Duration in years. Defaults to the seismic catalog duration.

``annualize``
    If ``True``, annualize the rates. Default: ``True``.


.. _gem-max-mag-check:

Maximum Magnitude Check (``max_mag_check``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A sanity check that verifies the model can produce earthquakes as large as the
largest observed earthquake in each spatial cell. Note that there can be issues
with very large earthquakes (with ruptures larger than the cell size), as the
hypocenter for an observed event may be in a different cell than the most
compatible source.

Parameters:

``append_check``
    Optional. Boolean. If ``True``, append pass/fail results to the bin data.

``warn``
    Optional. Boolean. If ``True``, log warnings for each failing cell.


Other Evaluations
-----------------

.. _gem-moment-over-under-eval:

Moment Over/Under Evaluation (``moment_over_under``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generates many stochastic catalogs and compares the total seismic moment release
in each spatial cell to the observed moment release. This helps highlight areas
that are more or less seismically productive than the observations support.

Parameters:

``investigation_time``
    Duration of the catalog in years.

``n_iters``
    Number of stochastic event sets to generate.

``min_mag``
    Optional. Minimum magnitude for moment calculation. Defaults to
    ``mfd_bin_min``.

``max_mag``
    Optional. Maximum magnitude for moment calculation. Defaults to
    ``mfd_bin_max``.


.. _gem-rupture-matching-eval:

Rupture Matching Evaluation (``rupture_matching_eval``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Matches observed earthquakes to modeled ruptures based on proximity, magnitude
similarity, and (optionally) geometric similarity (attitude and rake). This
evaluation helps assess whether the model contains ruptures that are consistent
with the observed earthquakes.

Parameters (all optional, with defaults):

``use_occurrence_rate``
    Weight matches by rupture occurrence rate. Default: ``False``.

``distance_lambda``
    Distance decay parameter for the matching function. Default: ``1.0``.

``mag_window``
    Magnitude window for considering candidate ruptures. Default: ``1.0``.

``group_return_threshold``
    Threshold for group matching. Default: ``0.9``.

``min_likelihood``
    Minimum match likelihood. Default: ``0.1``.

``no_attitude_default_like``
    Default likelihood when a rupture has no attitude data. Default: ``0.5``.

``no_rake_default_like``
    Default likelihood when a rupture has no rake data. Default: ``0.5``.

``return_one``
    ``"best"`` to return only the best match, or ``"all"`` for all matches
    above the threshold. Default: ``"best"``.

``parallel``
    Use parallel processing for matching. Default: ``False``.


.. _gem-cumulative-occurrence-eval:

Cumulative Occurrence Evaluation (``cumulative_occurrence_eval``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Evaluates the cumulative earthquake occurrence over time for each magnitude bin,
comparing the observed temporal pattern to the model's predicted rate. This is
useful for identifying temporal clustering or quiescence relative to the model.

Takes no configuration parameters (use ``{}``).


.. _gem-catalog-ground-motion-eval:

Catalog Ground Motion Evaluation (``catalog_ground_motion_eval``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compares observed ground motions from a flatfile with model predictions. This
requires a flatfile to be specified under ``input.flatfile`` in the
configuration.

Parameters:

``match_rups``
    Optional. If ``True``, match ruptures to earthquakes before computing
    ground motion comparisons. Default: ``False``.

``gmf_method``
    Optional. Method for ground motion calculation. Default:
    ``"ground_motion_fields"``.
