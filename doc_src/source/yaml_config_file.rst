.. _yaml-config:


===========================
The YAML configuration file
===========================

A Hamlet evaluation is configured using a `YAML <https://yaml.org>`_ file,
which describes the tests and evaluations that should be run and any required parameters, the
location of the input files, and what outputs should be written. The file is
made into a dictionary that is used to guide the Hamlet run, so the order
of items is not important.

A simple example of a YAML configuration file is given at the bottom of this
page, and a comprehensive reference configuration (with comments explaining
every option) is available in the source repository at
``doc_src/source/example_configs/reference_config.yml``.


.. _metadata:

Metadata (``meta``)
===================

This optional section describes the model. The information is used when
writing reports.

.. code-block:: yaml

    meta:
        model_name: Model Name
        description: "words describing the model"



.. _eval-config:

Test and Evaluation Configuration (``config``)
==============================================

This section specifies which frameworks, tests, and evaluations to run, along
with their parameters.


``model_framework``
-------------------

This is a dictionary of frameworks to run. Each framework key contains
its own tests and evaluations as sub-keys. Multiple frameworks can be run in a
single Hamlet run. Available frameworks are:

* ``gem`` -- GEM tests and evaluations (see :ref:`gem-tests-evaluations`). The
  most supported and/or configurable versions of all tests and evaluations are 
  available here. It is recommended to use these.
* ``relm`` -- RELM/CSEP tests (see :ref:`relm-tests`). These 'letter tests' are 
  found with specific defaults set.
* ``sanity`` -- Basic sanity checks (see :ref:`sanity-checks`).

Tests and evaluations are nested directly under their framework, with their
configuration parameters as sub-keys. Use an empty dictionary (``{}``) for
those that need no configuration.

.. code-block:: yaml

    config:
      model_framework:
        gem:
          model_mfd: {}
          max_mag_check:
            append_check: True
            warn: True
          N_test:
            conf_interval: 0.95
            prob_model: poisson
          M_test:
            critical_frac: 0.025
            n_iters: 1000
        relm:
          S_test:
            critical_frac: 0.25
            n_iters: 1000
            investigation_time: 40.


``parallel``
    A Boolean (``True`` or ``False``) flag that determines whether
    parallel algorithms are used for loading the seismic source model and
    performing the more time-intensive tests and evaluations (such as Monte
    Carlo based consistency tests and moment analysis).

    This flag should be ``True`` for medium to large models, unless RAM is a
    major limitation. For small models, it may be faster to run on a single
    core, because the overhead in instantiating multiple processes can be
    substantial.

``rand_seed``
    An optional integer `random seed`_ for reproducible random sampling (which
    is fundamental in Hamlet, as stochastic catalogs are generated for most
    tests and evaluations).

``log_file``
    An optional path to a log file. If given, log output is written to this
    file in addition to the console.

.. _random seed: https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.seed.html


.. _input:

Inputs (``input``)
==================

This section describes the inputs into a Hamlet run: the seismic source model,
the observed earthquake catalog, and the configuration of the spatial bins.


Seismic Source Model (``ssm``)
------------------------------

The SSM must be in the modern OpenQuake format. It can be specified either with
a logic tree XML file or an OpenQuake job.ini file. All SSM parameters default
to ``null`` when not specified (see ``cfg_defaults`` in
``openquake.hme.core.core``).

Sub-parameters:

``ssm_dir``
    Filepath (absolute or relative) to the directory containing the seismic
    source model logic tree file.

``ssm_lt_file``
    The name of the logic tree XML file for the SSM, i.e. ``ssmLT.xml``
    for most GEM Hazard Mosaic model repositories.

``job_ini_file``
    Alternative to ``ssm_dir`` + ``ssm_lt_file``: path to an OpenQuake
    ``job.ini`` file that specifies the source model. This is useful when
    the source model configuration is complex.

``branch``
    Specifies which logic tree branch to evaluate. Because different branches
    are often mutually-exclusive, alternative descriptions of earthquake
    occurrence, evaluating multiple branches at once may greatly increase the
    forecasted occurrence rates and result in inaccurate evaluation.

    Set to ``"iterate"`` to evaluate each branch independently in a single
    Hamlet run. This loads the earthquake catalog once and then evaluates each
    branch separately, producing combined results.

``tectonic_region_types``
    Optional filter specifying which Tectonic Region Type(s) should be
    evaluated. The types must correspond to those in the SSM. Pass as a YAML
    list or omit/set to ``null`` to include all types.

    .. code-block:: yaml

        tectonic_region_types:
            - Active Shallow Crust
            - Stable Continental

``source_types``
    Optional filter specifying which source types to evaluate. Options include
    ``simple_fault``, ``complex_fault``, ``area``, ``point``, ``multipoint``,
    and ``MultiFaultSource``. Pass as a YAML list, or ``null`` for all types.


Depth Filtering (Optional)
--------------------------

``min_depth``
    Optional. Minimum source depth in km. Sources shallower than this are
    excluded from the evaluation. Default: ``0.0``. Specified at the
    ``input`` level (not inside ``ssm``).

``max_depth``
    Optional. Maximum source depth in km. Sources deeper than this are
    excluded from the evaluation. Default: no limit. Specified at the
    ``input`` level (not inside ``ssm``).

    .. code-block:: yaml

        input:
          min_depth: 0
          max_depth: 40


Observed Earthquake Catalog (``seis_catalog``)
----------------------------------------------

This set of parameters determines how the seismic catalog will be found and
parsed so that it can be compared to the source model.

``seis_catalog_file``
    Relative or absolute filepath to the CSV earthquake catalog file.

Temporal Parameters
~~~~~~~~~~~~~~~~~~~

Provide any two of ``start_date``, ``stop_date``, and ``duration``; the third
will be calculated automatically. Alternatively, use a ``completeness_table``.

``start_date``
    Start of the catalog time window. Can be an integer (interpreted as
    January 1 of that year) or a date string (e.g. ``"1976-01-01"``).

``stop_date``
    End of the catalog time window. Same format as ``start_date``.

``duration``
    Duration of the catalog in years (float).

``completeness_table``
    A list of ``[year, magnitude]`` pairs defining the completeness threshold
    over time. Each pair means "the catalog is complete above this magnitude
    from this year onward." When used, the effective duration varies by
    magnitude bin, which enables more accurate evaluation of models across
    the magnitude range.

    .. code-block:: yaml

        completeness_table:
          - [1960, 5.0]
          - [1900, 7.2]

Column Mappings (``columns``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maps expected fields to actual column names in the CSV file. Only specify
columns whose names differ from the defaults.

* ``x_col`` -- Defaults to ``longitude``.
* ``y_col`` -- Defaults to ``latitude``.
* ``depth`` -- Defaults to ``depth``.
* ``magnitude`` -- Defaults to ``magnitude``.
* ``source`` -- No default. The institutional source of the earthquake
  (e.g. ``Agency``).
* ``event_id`` -- No default. The earthquake ID column (e.g. ``eventID``).
* ``time`` -- Either a single column name containing a parseable timestamp,
  or an ordered list of columns to construct the time:

  .. code-block:: yaml

      time:
          - year
          - month
          - day
          - hour
          - minute
          - second


Bins (``bins``)
---------------

This section configures the spatial and magnitude binning used throughout
Hamlet.

Default values for many parameters (including ``h3_res``, ``simple_ruptures``,
``subset``, and ``rupture_file`` options) are defined in
``openquake.hme.core.core.cfg_defaults`` and are applied automatically when
not specified in the YAML file.

``h3_res``
    The resolution of the `H3 <https://github.com/uber/h3-py>`_ hexagonal
    grid used for spatial binning. Values range from 0 (coarsest, ~4,000 km
    edge length) to 15 (finest). Default: ``3`` (~69 km edge length).
    H3 hexagons are generated automatically based on the spatial extent of
    the source model.

``mfd_bin_min``
    Minimum magnitude for the MFD. Required.

``mfd_bin_max``
    Maximum magnitude for the MFD. Required.

``mfd_bin_width``
    Width of the magnitude bins. Required.


Rupture File Caching (``rupture_file``)
---------------------------------------

Loading ruptures from a source model can be slow for large models. These
optional parameters let you save processed ruptures to disk and reload them
in subsequent runs, if you have not changed the model and are just changing the
test configuration (branches, depth ranges, catalog completeness, etc.).

.. code-block:: yaml

    rupture_file:
      read_rupture_file: false
      save_rupture_file: false
      rupture_file_path: ./ruptures.hdf5

``read_rupture_file``
    If ``true``, read ruptures from the file at ``rupture_file_path`` instead
    of processing the SSM. Default: ``false``.

``save_rupture_file``
    If ``true``, save processed ruptures to ``rupture_file_path``. Default:
    ``false``.

``rupture_file_path``
    Path to the rupture file. Supported formats: ``.hdf5``, ``.feather``,
    ``.csv``.


Spatial Subsetting (``subset``)
-------------------------------

Optional parameters to restrict the evaluation to a geographic subset of
the model domain.

.. code-block:: yaml

    subset:
      file: path/to/subset.geojson
      buffer: 0.0

``file``
    Path to a GIS file containing the subset geometry. Default: ``null``.

``buffer``
    Buffer distance around the subset geometry (in the units of the GIS file's
    coordinate reference system). Default: ``0.0``.


Flatfile (``flatfile``)
-----------------------

Path to a ground motion flatfile (CSV). Required for the
``catalog_ground_motion_eval`` evaluation. Specified at the ``input`` level.

.. code-block:: yaml

    input:
      flatfile: path/to/flatfile.csv


Note that this requires a flatfile that is constructed with a format that is
used internally at GEM (and currently undocumented, pending publication). Until
the format stabilizes, there will almost certainly be problems.


Other Input Options
-------------------

``simple_ruptures``
    Boolean. Use simplified rupture representations (hypocenter point sources) for faster processing. Default: ``True``.


.. _report:

Reporting (``report``)
======================

This optional section configures report generation. Currently there is one
option, the ``basic`` HTML report, which aggregates all test and evaluation
results into maps, plots, and tables.

.. code-block:: yaml

    report:
      basic:
        outfile: outputs/report.html


.. _output:

JSON Output (``json``)
======================

Optional. Write test results as a JSON file.

.. code-block:: yaml

    json:
      outfile: outputs/results.json


Minimal Example
===============

.. literalinclude:: example_configs/minimal_config.yml
    :language: yaml


Comprehensive Reference
=======================

The complete reference configuration with all options documented is available
at ``doc_src/source/example_configs/reference_config.yml`` in the repository.

.. literalinclude:: example_configs/reference_config.yml
    :language: yaml
