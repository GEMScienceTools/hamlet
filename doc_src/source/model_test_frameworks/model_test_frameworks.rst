========================
Model Testing Frameworks
========================

Hamlet provides four test frameworks that can be used independently or combined
in a single evaluation. Each framework is specified as a key under
``config.model_framework`` in the YAML configuration file.


.. include:: ./gem_tests.rst


.. _relm-tests:

RELM/CSEP Tests
================

The RELM_ (Regional Earthquake Likelihood Models) / CSEP tests are
implementations of the standard CSEP consistency tests. These are similar to the
GEM tests but differ in some statistical assumptions -- notably,
``not_modeled_likelihood`` is hardcoded to ``0.0`` in the RELM framework,
meaning that any observations in unmodeled cells or magnitude bins will cause
the test to fail.

Available tests:

.. _relm-N-test:

N-Test
------

Compares the total number of observed earthquakes to the model prediction.
See :ref:`gem-N-test` for details on the test logic.

Parameters:

``conf_interval``
    Confidence interval for the test (e.g. ``0.95``).

``prob_model``
    Probability model: ``"poisson"`` or ``"poisson_cum"``.

``investigation_time``
    Duration in years.

``prospective``
    Optional. Use prospective catalog. Default: ``False``.


.. _relm-M-test:

M-Test
------

Evaluates the consistency of the magnitude-frequency distribution.
See :ref:`gem-M-test` for details on the test logic.

Parameters:

``critical_frac``
    Fraction threshold for test failure (e.g. ``0.25``).

``n_iters``
    Number of Monte Carlo iterations.

``investigation_time``
    Duration in years.

``prospective``
    Optional. Use prospective catalog. Default: ``False``.


.. _relm-S-test:

S-Test
------

Evaluates the spatial consistency of the model.
See :ref:`gem-S-test` for details on the test logic.

Parameters:

``critical_frac``
    Fraction threshold for test failure.

``n_iters``
    Number of Monte Carlo iterations.

``investigation_time``
    Duration in years.

``prospective``
    Optional. Default: ``False``.

``likelihood_function``
    Optional. ``"mfd"`` or ``"conf_interval_poisson"``. Default: ``"mfd"``.

``normalize_n_eqs``
    Optional. Normalize by number of earthquakes. Default: ``False``.


.. _relm-L-test:

L-Test
------

Joint likelihood test combining spatial and magnitude information.

Parameters:

``critical_frac``
    Fraction threshold for test failure.

``n_iters``
    Number of Monte Carlo iterations.

``investigation_time``
    Duration in years.

``prospective``
    Optional. Default: ``False``.


.. _sanity-checks:

Sanity Checks
=============

Sanity checks are basic evaluations to verify that the model is internally
consistent and matches the observations at a gross level.

.. _max-mag-check:

Maximum Magnitude Check (``max_check``)
---------------------------------------

Evaluates the observed seismicity and MFD inside each spatial cell to check
whether the maximum magnitude of the MFD is larger than the largest observed
earthquake. Cells where the observed maximum magnitude exceeds the model
maximum magnitude are flagged.

Parameters:

``warn``
    Optional. Boolean. If ``True``, log a warning for each cell that fails
    the check. Default: ``True``.


.. _model-description:

Model Description
=================

The model description framework provides characterization and descriptive
analysis of the model. These are not pass/fail tests, but rather produce
summaries and comparisons useful for understanding the model.

``describe_mfds``
-----------------

Computes and compares the model and observed magnitude-frequency distributions,
extending the magnitude range to fully encompass the data. Takes no
configuration parameters (use ``{}``).

``describe_model``
------------------

General model characterization, which currently includes the MFD description.
Takes no configuration parameters (use ``{}``).


.. _RELM: http://cseptesting.org/documents/relm.php
