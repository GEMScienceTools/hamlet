==================================================
Hamlet: Hazard Model Evaluation and Testing
==================================================

Hamlet (``openquake.hme``) is a Python package developed (OK, in development)
for qualitative and quantitative evaluation of Probabilistic Seismic Hazard
Analysis (PSHA) models, with the intention of providing feedback to modelers
during the model construction process, to aid model development. Hamlet is
developed by the `GEM Foundation`_, and uses the OpenQuake_ software
extensively.

Hamlet will incorporate several model test frameworks, including those
developed by GEM and some of those developed outside of GEM such as the RELM_
tests. Currently, and likely in the future, the model files will be required to
be in the OpenQuake_ format, regardless of the format of their original
implementation.

Most of the Hamlet evaluations are spatial in nature; the model domain is
discretized into grid cells, and comparisons between observations and model
predictions are performed in each grid cell, to highlight where in the domain
the model matches the observations, and where it might need some refinement.

Additionally, unlike some other hazard model testing frameworks, Hamlet is
designed to operate on separate components of a hazard model, so that each
component can be evaluated against its corresponding data. For example, each
branch of a source model logic tree can be tested independently, and each type
of source (e.g., subduction megathrust, crustal, in-slab) can be tested
independently as well, in the spatial framework described above.

Quickstart
==========

Installation
------------

Hamlet requires installation Python v.3.7+, the OpenQuake_ engine, and some
additional dependencies as well. These are specified in the ``requirements.txt``
file.

(*Note:* A few of the dependencies might be challenging to install.  These are
Rtree_ and h3-py_. You may have to install ``libspatialindex`` or
``libspatialindex-dev`` on Linux or MacOS first, depending on your system, for
``Rtree``.  ``h3-py`` requires ``cc`` and ``make``, but then on Linux/MacOS can
be installed easily. Please see the documentation for each.)

First, install the OpenQuake_ engine, following directions on that website. You
probably want to install it into a virtual environment, and you may even want to
have a separate virtual environment for running Hamlet than the OpenQuake_
virtual environment that you normally use (this is up to you).

Then, clone the Hamlet repository, and from that directory, install the
requirements::

    pip install -r requirements.txt

and then install Hamlet::

    pip install -e .


Running Hamlet
--------------

Hamlet only requires a seismic hazard model (implemented in OpenQuake_) and a
processed seismic catalog (declustered, and ideally classified by source type)
to run. Once installed, Hamlet can be run from the command line::

    hamlet test_model.yml

``test_model.yml`` is a YAML_ configuration file that specifies the source
model, seismic catalog, tests to be run, and other variables and parameters.


Documentation
=============
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   architecture
   model_test_frameworks/model_test_frameworks



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _YAML: https://yaml.org
.. _GEM Foundation: https://www.globalquakemodel.org
.. _OpenQuake: https://github.com/GEM/oq-engine
.. _RELM: http://cseptesting.org/documents/relm.php
.. _h3-py: https://github.com/uber/h3-py
.. _Rtree: https://toblerity.org/rtree/
