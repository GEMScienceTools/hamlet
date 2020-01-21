.. _yaml-config:


===========================
The YAML configuration file
===========================

A Hamlet evaluation is configured using a `YAML <https://yaml.org>`_  file,
which describes the tests that should be run and any required parameters, the
location of the input files, and what outputs should be written.  The file is
made into a dictionary that is used guide the Hamlet run. Therefore the order
of items is not important.


A simple example of a yaml configuration file is given at the bottom of this
page.

Here are the components of the YAML config file:


.. _metadata:

Metadata (``meta``)
===================

This is a basic section that describes the model. The information is used when
writing reports; otherwise it is not necessary.

.. code-block:: yaml

    meta:
        model_name: Model Name
        description: "words describing the model name"



.. _eval-config:

Test and Evaluation Configuration (``config``)
==============================================

This section lists the different tests or evaluations to be run, and what the
parameters for each will be.


``model_framework``
    This specifies which framework to use (see
    :doc:`./model_test_frameworks/model_test_frameworks` for more information on
    these). Currently, only the ``gem`` option is available, but ``sanity`` and
    ``relm`` are close to being usable. However the ``sanity`` checks are
    available through the ``gem`` framework as well.

``parallel``
    This is a Boolean (``True`` or ``False``) flag that determines whether
    parallel algorithms are used for loading the seismic source model, and
    performing the more time-intensive tests (such as Monte Carlo based tests).
    The parallel algorithms produce equivalent results, using Python's
    `multiprocessing <https://docs.python.org/3.7/library/multiprocessing.html>`_
    facilities.

    This flag should be ``True`` for medium to large models, unless RAM is a
    major limitation. For small models, it may be faster to run on a single
    core, because the overhead in instantiating multiple processes, and
    processing the results, can be substantial.

``rand_seed``
    A `random seed`_ to be used for reproducible Monte Carlo simulations and
    other functions using random sampling. The seed must follow Numpy's rules;
    to keep it simple, just use an integer.  Optional.

.. _random seed: https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.seed.html

Tests
-----

``tests``
    This specifies which tests or evaluations within the
    ``model_test_framework`` should be run. Multiple tests can be run; these
    should be nested below the ``tests`` item with their configuration variables
    given. 

    For the ``gem`` framework, the available tests are 
    :ref:`likelihood <gem-like-test>`,
    :ref:`model_mfd <gem-model-mfd-test>`, and 
    :ref:`max_mag_check <max-mag-check>`.


An example, illustrating the required nesting:

.. code-block:: yaml

    config:
        model_framework: gem
        tests:
            likelihood:
                likelihood_method: poisson
                investigation_time: 40.
            model_mfd:
                investigation_time: 40.
            max_mag_check:
                append_check: True
                warn: True
        parallel: False
        rand_seed: 420


.. _input:

Inputs (``input``)
==================

This section describes the inputs into a Hamlet run: the seismic source model,
the observed earthquake catalog that is used to evaluate the seismic source
model, and the configuration of the spatial bins that will hold the SSM.


Seismic Source Model (``ssm``)
------------------------------

This section describes the location of the Seismic Souce Model (SSM), and the
components of the model that should be evaluated with Hamlet.

The SSM must be in the modern OpenQuake format, and specified with a logic tree
XML file.

Sub-parameters:

* ``ssm_dir``
    This parameter describes the filepath (either absolute or relative) to the
    directory containing the seismic source model logic tree file.

* ``ssm_lt_file``
    This is the name of the logic tree XML file for the SSM, i.e. ``ssmLT.xml``
    for most GEM Hazard Mosaic model repositories.

* ``branch``
    This specifies what branch is to be evaluated. Because different branches
    are often mutually-exclusive, alternative descriptions of earthquake
    occurrence, evaluating multiple branches at once may greatly increase the
    forecasted occurrence rates of earthquakes, and result in highly innacurate
    evaluation of the model.

* ``tectonic_region_types``
    This parameter is optional but specifies which Tectonic Region Type(s)
    should be evaluated. For example, one may want to evaluate different region
    types separately, to better understand how well the model predicts
    seismicity in active crust vs. stable regions.  The types must correspond to
    those in the SSM.

    This can be specified as a list, or by passing ``null`` to consider all
    tectonic region types.  For example:

.. code-block:: yaml

    tectonic_region_types:
        - Active Shallow Crust
        - Stable Continental


* ``source_types``
    This parameter is optional but specifies which source types are evaluated.
    Options include ``simple_fault``, ``complex_fault``, ``area``, ``point``,
    and ``multipoint``.  As with the ``tectonic_region_types``, the values need
    to be passed as a YAML list, or the value of ``null`` should be given.
    A single value may be passed as

.. code-block:: yaml

    source_types:
        - point



Observed earthquake catalog (``seis_catalog``)
----------------------------------------------

This set of parameters determines how the seismic catalog will be found and
parsed so that it can be compared to the source model.

``seis_catalog_file``
    This parameter gives the relative or absolute filepath and filename of the
    CSV file that is the earthquake catalog.

``columns``
    This parameter gives a list of the expected column names with the names of
    the columns in the CSV file. The first set of column names are optional,
    with very common defaults, and only need to be put into the YAML file
    if the names to do not correspond to the defaults.  If the earthquake
    catalog is made from the ISC-GEM or the GEM toolkits, these columns
    do not need to be given in the YAML file:

    * ``x_col``
        This defaults to ``longitude``.  If this column in the CSV is not
        called ``longitude``, please pass the column name.

    * ``y_col``
        This defaults to ``latitude``.  If this is not the column name in the
        CSV file, please pass the column name.

    * ``depth``
        This defaults to ``depth``. If this is not the column name in the
        CSV file, please pass the column name.

    * ``magnitude``
        This defaults to ``magnitude``. If this is not the column name in the
        CSV file, please pass the column name.

    The following columns have no defaults; if they are to be used, they should
    be set here:

    * ``source``
        This column specifies the institutional source of the earthquake, not
        the geological source.  In the GEM catalogs, this is commonly
        ``Agency``.

    * ``event_id``
        This is the ID of the earthquake.  In the GEM catalogs, this is
        commonly ``eventID``.

    And finally, a more complex column, which is actually a set of columns in
    many seismic catalogs:

    * ``time``
        This sub-parameter is an ordered list of the columns specifying the time
        components that are used to make a single time for the earthquake (this
        is done by making a :class:`datetime.datetime` time object; please see
        `dateutil <https://dateutil.readthedocs.io/en/stable/>`_ for more
        information). For the typical GEM catalog, the time sub-parameter will
        be specified like this:

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
This section describes configuration for the
:class:`~openquake.hme.utils.bins.SpacemagBin` s that are used throughout Hamlet.

All of the parameters are optional, and should only be listed in the YAML file
if the defaults are not suitable.

Sub-parameters:

* ``bin_gis_file``
    Optional path to a GIS file with polygons that will become the bins. The GIS
    file will be read with `GeoPandas <geopandas.org>`_ and one can find
    information on the acceptable vector GIS filetypes at that link.

    It is not recommended to use this option. If it is not given, then polygons
    are created using Uber's `h3-py <https://github.com/uber/h3-py>`_ library
    automatically (based on the spatial extent of the seismic source model),
    which results in much faster testing and is far less of a hassle than
    creating your own GIS file. However if the polygons should correspond to
    e.g. seismic source zones for some purpose, then this option could be used.

* ``mfd_bin_max``
    Maximum size of the model MFD to be considered. Defaults to 9.0.

* ``mfd_bin_min``
    Minimum size of the model MFD to be considered. Defaults to 6.0.

* ``mfd_bin_width``
    Width of the bins for the MFD. Defaults to 0.2. Narrower bins may have
    issues with lower-resolution source models or seismic catalogs.

.. _report:

Reporting
=========

This optional section describes how reports (listing the results of the
evaluations) will be written. Currently there is only one option, the ``basic``
HTML report. This report will aggregate all of the tests or evaluations
performed and summarize them, either in a map, plot, or table.

It is suggested to use this option instead of writing plots directly
either through the ``model_mfd`` test configuration, or the ``output``
configuration below, as it is a nicer summary.  However if one has specific
goals, then those other options may be used.

To write the basic HTML report, put this section in the YAML:

.. code-block:: yaml

    report:
        basic:
            outfile: /path/to/file.html

In the future, more types of reports may be generated, but this is sufficient
for now.



.. _output:

Outputs
=======

This optional section specifies writing other outputs than reports (plots or GIS
files). This option can give more fine-grained control over the outputs than the
``report`` option, at this time. However the results are less convenient than
writing a report.

Parameters:

``bin_gdf``
    If this parameter is passed, a GeoJSON file will be written that contains
    much of the information in the Hamlet evaluation, including test results,
    for each spatial bin.  The filename to be used is given with a sub-parameter
    ``file``:

.. code-block:: yaml

    bin_gdf:
        file: /path/to/file.geojson


``plots``
    This is a parameter (that may be soon deprecated) describing how an MFD plot
    should be written.  It can offer more control than the other options,
    but is not not thoroughly documented at this time.


Putting it all together
=======================

A simple example is taken from the test suite below:

.. literalinclude:: ../../tests/model_test_frameworks/gem/integration_tests/gem_sm1/test_sm1_poisson.yml
    :language: yaml
