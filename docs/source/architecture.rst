########################################
Hamlet Architecture and Testing Workflow
########################################

Currently, Hamlet performs spatial-temporal hazard model checks and
statistical evaluation of model consistency and performance against an observed
earthquake catalog.

Hamlet follows this work process:

1. Read in YAML configuration file, that specifies:

   - Which tests to be run

     - What parameters for each test

   - What input files:

     - Seismic Source Model files

     - Seismic catalog (observed earthquakes)

     - GIS file of spatial bins

   - What to output:

     - GIS files, etc.

2. Reads and process SSM:

   1. Loads sources from a single logic tree branch

   2. Sorts sources based on their type, with a list for each

3. Sorts the ruptures from all sources by magnitude and into spatial bins:

   - Makes :class:`openquake.hme.utils.bins.SpacemagBin` class that holds
     ruptures, observed earthquakes, and both model and empirical
     Magnitude-Frequency distributions for each bin.

4. Runs the tests:

   - Basic sanity checks (e.g., whether the observed earthquake maximum
     magnitude exceeds the model maximum magnitude in each spatial bin)

   - Statistical evaluation (i.e., model likelihoods based on the calculated
     probabilities of observing the earthquakes in a catalog given the SSM)

   - Multiple tests can be run sequentially, without reloading the SSM.

5. Print/write output.

  - GIS files with the test results for each bin

  - CSV files of the total model and catalog (within the source bins) MFDs


At this pre-release stage, most of the development has focused on writing the
test framework, rather than creating a broad suite of tests. However, the
framework is functional at this point, and the development of a test suite is
the next priority.
