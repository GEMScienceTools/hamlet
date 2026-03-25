========================
Model Testing Frameworks
========================


.. include:: ./gem_tests.rst

Sanity Checks
=============

Sanity checks are small evaluations to make sure that basic parameters of the
model are internally consistent and match the observations at a gross level. An
example is making sure that the MFDs in each grid cell are capable of producing
earthquakes as large as the largest observed earthquakes.

Tests
-----

.. _max-mag-check:

*Maximum Magnitude Check*

The Maximum Magnitude Check evaluates the observed seismicity and MFD inside
each class:`~openquake.hme.utils.bins.SpacemagBin` to see whether the maximum
magnitude of the MFD is larger than the observed earthquakes.

Parameters:

``append_check``
    This is a Boolean (``True`` or ``False``).  If ``True``, the row for each
    :class:`~openquake.hme.utils.bins.SpacemagBin` in the dataframe used to
    store everything will have a column appended that gives ``True`` and
    ``False`` values for if the test passes or not (``True`` is pass).

``warn``
    This is a Boolean (``True`` or ``False``) that determines whether a
    warning is given on the command line (or in the log file) for each
    class:`~openquake.hme.utils.bins.SpacemagBin` that fails the test.



RELM Tests
==========

The RELM_ (Regional Earthquake Likelihood Model) tests are quite similar in
principle to the GEM Magnitude-Frequency evaluations, though there are some
differences in the statistical assumptions used to aggregate the likelihood
statistics from model sub-regions to the model as a whole.

.. _N_test:

*N test*

    The N-test compares the total number of events in the catalog to the total
    number of events in stochastic events sets generated from the source model.
    This is the count of the lowest-magnitude bin or point in the
    magnitude-frequency distribution.

Parameters:


.. _M_test:

*M test*

.. _S_test:


.. _RELM: http://cseptesting.org/documents/relm.php