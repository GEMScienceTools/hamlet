.. _gem-tests-evaluations:


GEM Tests and Evaluations
=========================


These are developed by GEM, sometimes based on the literature, or sometimes
based on our own ideas and implementations. See for
more information, or :mod:`~openquake.hme.model_test_frameworks.gem` for the
function documentation. Sanity checks (as detailed below) are also available
from the GEM testing framework, for convenience during the workflow.

Magnitude-Frequency Distribution based tests
--------------------------------------------

These tests evaluate a model based on the magnitude-frequency distribution (MFD)
inside each grid cell, or in the model as a whole, (for the model component
being tested).


.. _gem-like-test:

*Likelihood test*

Currently, there are two implementations of the MFD likelihood tests, one
'empirical', based on Monte Carlo sampling of the source model in each bin, and
one based on the frequencies in the MFD itself.

In both of these tests, the likelihood of observing the seismicity in the
catalog given the model MFD is calculated through the 

Parameters (all are optional, as default values are supplied):

``likelihood_method``
    This is how the computations are performed.  ``poisson`` uses the Poisson
    likelihood, and ``empirical`` uses a Monte Carlo sampling of the MFD.  The
    default value is ``poisson``.

``investigation_time``
    This is how long the time period is for comparing the observed seismicity to
    the MFD. Unless you're doing something very crafty, the value should be in
    years, and should be the length of the observed earthquake catalog.  In the
    future, a completeness table may be used instead of this parameter.  The
    default value is ``1.``

``default_likelihood``
    This is the likelihood that results if no earthquake sources are present in
    the grid cell. If the cells are built using `h3` (the default option, if no
    GIS file for the test is supplied), this parameter will have no effect.  The
    default value is ``1.``


``not_modeled_val``
    This is the likelihood that results if the rate of earthquake production in
    that magnitude bin is zero, but there are earthquakes within the magnitude
    bin. Standard (or naive) statistical theory suggests that this value should
    be zero, as it is in the RELM tests, but because this value is multiplied by
    all of the other magnitude bins inside each spatial bin/grid cell, a single
    zero value will make the whole model likelihood zero.  The default is
    ``1e-5`` which is a bit more pragmatic.



.. _gem-model-mfd-test:

*Model MFD Evaluation*

The Model MFD evaluation sums up the MFDs from each
:class:`~openquake.hme.utils.bins.SpacemagBin` and makes an MFD for the whole
model, which is then compared to the observed MFD from the earthquake catalog.
This is to produce a figure and currently does not yield any quantitative values
or evaluate the entire model goodness of fit.

Parameters (all are optional):

``investigation_time``
    This is the duration of the comparison between the observed seismicity to
    the MFD. See :ref:`likelihood <gem-like-test>` above for more information.

``out_csv``
    This parameter specifies a filename. If this is given, a CSV table of the
    observed and modeled MFDs will be written.

``out_plot``
    This parameter specifies a filename. If this is given, a plot of the
    observed and modeled MFDs will be written. The file suffix will determine
    the plot format.  Common formats include ``png``, ``svg`` and ``pdf``.  See
    the ``matplotlib`` docs for more info.


.._max_mag_check:

*Ensures that the model can produce the maximum observed seismcity in each cell*

This test is borrowed from the Sanity checks. It simply checks to see whether
the sources inside each cell are capable of producing earthquakes as large as
the largest observed earthquakes. Note that there can be some issues with very
large earthquakes (with ruptures larger than the cell size) as the hypocenter
for an observed event may be in a different cell than the most compatible
hypocenter from the sources.


.. _gem-moment-over-under-eval:

*Compares observed and stochastic moment release*

This evaluation generates many synthetic catalogs (stochastic event sets) and
compares the total moment release in each cell for each of the catalogs to the
observed moment release. This evalution helps highlight areas that are more or
less seismically productive than the observations may support.

Parameters:

``investigation_time``:  Duration of the catalog (and of the generated
    stochastic event sets).

``n_iters``: Number of iterations (stochastic event sets) generated. Note that
    generating these catalogs is fairly time-intensive in the current
    implementation. For large models, the number of iterations should be kept to
    under 50-100 until performance is improved.

