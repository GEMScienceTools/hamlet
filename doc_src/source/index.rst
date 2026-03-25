==================================================
Hamlet: Hazard Model Evaluation and Testing
==================================================

Hamlet (``openquake.hme``) is a Python package for qualitative and quantitative
evaluation of Probabilistic Seismic Hazard Analysis (PSHA) models, with the
intention of providing feedback to modelers during the model construction
process, to aid model development. Hamlet is developed by the `GEM
Foundation`_, and uses the OpenQuake_ software extensively.

Hamlet incorporates several model test frameworks, including those developed by
GEM and some of those developed outside of GEM such as the RELM_/CSEP tests.
The model files must be in the OpenQuake_ format.

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

Hamlet requires Python 3.11+ and the OpenQuake_ engine.

First, install the OpenQuake_ engine, following directions on that website. You
probably want to install it into a virtual environment.

Then, clone the Hamlet repository, and from that directory, install the
requirements::

    pip install -r requirements.txt

and then install Hamlet::

    pip install -e .


Running Hamlet
--------------

Hamlet requires a seismic hazard model (implemented in OpenQuake_) and a
processed seismic catalog (declustered, and ideally classified by source type)
to run. Once installed, Hamlet can be run from the command line::

    hamlet test_model.yml

``test_model.yml`` is a :doc:`configuration file <yaml_config_file>` in YAML_
format that specifies the source model, seismic catalog, tests to be run, and
other variables and parameters.


Documentation
=============
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   yaml_config_file
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
