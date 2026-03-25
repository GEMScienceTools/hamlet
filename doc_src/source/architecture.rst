########################################
Hamlet Architecture and Testing Workflow
########################################

Hamlet performs spatial-temporal hazard model checks and statistical evaluation
of model consistency and performance against an observed earthquake catalog.

Hamlet follows this workflow:

1. Read in :doc:`YAML configuration file <./yaml_config_file>`, that specifies:

   - Which tests to be run, and their parameters

   - Input files:

     - Seismic Source Model files

     - Seismic catalog (observed earthquakes)

     - GIS file of spatial bins (optional)

   - What to output:

     - Reports

     - GIS files

     - JSON results

2. Read and process SSM:

   1. Load sources from a single logic tree branch (or iterate over all
      branches if ``branch: iterate`` is set)

   2. Sort sources based on their type

3. Sort the ruptures from all sources by magnitude and into spatial bins:

   - Uses `H3 <https://github.com/uber/h3-py>`_ hexagonal cells (by default)
     to spatially bin ruptures and observed earthquakes

   - Computes both model and empirical Magnitude-Frequency Distributions (MFDs)
     for each spatial bin

4. Run the tests:

   - Basic sanity checks (e.g., whether the observed earthquake maximum
     magnitude exceeds the model maximum magnitude in each spatial bin)

   - Statistical evaluation (N-test, M-test, S-test, L-test, and others,
     evaluating count, magnitude, spatial, and likelihood consistency)

   - Model characterization (MFD comparisons, moment rate analysis, rupture
     matching, ground motion evaluation)

   - Multiple tests can be run sequentially, and multiple frameworks can be
     used in a single run

5. Write output:

   - HTML reports summarizing the results with maps, plots, and tables

   - GIS files (GeoJSON) with the test results for each spatial bin

   - JSON files with structured test results


Branch Iteration
================

When ``branch: iterate`` is set in the configuration, Hamlet evaluates each
logic tree branch independently in a single run:

1. The earthquake catalog is loaded once (shared across all branches)

2. The complete source model is loaded, and all branches are extracted

3. For each branch:

   a. Ruptures are processed for that branch only

   b. Input data is grouped into spatial bins

   c. All configured tests are run

   d. Results are collected and memory is freed

4. Results from all branches are combined into a single report
