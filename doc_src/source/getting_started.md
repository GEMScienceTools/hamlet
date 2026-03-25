# Getting started with Hamlet

Hamlet is easy to use once the input data and configuration are prepared. The
steps to do this are:

## Prepare the hazard model

The hazard model must be in the OpenQuake format. There must be either a single
XML file that describes the seismic source model logic tree (including the
locations of the source XML files and other logic tree parameters), or an
OpenQuake `job.ini` file.

### Decide how the model should be evaluated

Choose whether Hamlet should be run for the entire model as a whole, or for
different components of the model (different logic tree branches, different
seismic source types, etc.). This will control how the data preparation and
testing are done.

A single branch can be evaluated by specifying the `branch` parameter, or all
branches can be evaluated independently in one run by setting `branch: iterate`.

### (Optional) Organize the hazard model with a `hamlet` directory

It is recommended to add a new `hamlet` directory with `data` and `output`
sub-directories.

The `data` directory can hold the seismic catalogs and, if
necessary, GIS files specifying the model domain and grid cells.

The `output` directory will hold the HTML reports and any other outputs that are
written during the testing procedure.

The [YAML configuration files](yaml_config_file.html) can be placed in the main `hamlet` directory:

```
model/
    in/
        ssm/
        gmm/
    out/
    hamlet/
        data/
            crustal_catalog.csv
            slab_catalog.csv
            full_catalog.csv (unused in testing)
        output/
        test_ssm_crustal.yml
        test_ssm_slab.yml
```

However, the user is free to organize the Hamlet files in any way; there is no
requirement that the files are in the same directory as the source model, as
long as the paths to the source model and earthquake catalogs are correct in the
YAML configuration file.

### (Optional) Make a new `git` branch for Hamlet

This may be a good way of organizing the results and running Hamlet in a
continuous integration system.

For example, a `git` branch called `hamlet` can have a separate `hamlet`
directory, as specified above. This directory does not exist in the `master`
branch or other branches, and when changes are made to those branches, they can
be pulled into the `hamlet` branch and be evaluated.

This can be configured in a continuous integration environment such that
Hamlet is run on the CI server when the `hamlet` branch is committed, but not
necessarily when commits to `master` or development branches are made.


## Prepare the earthquake catalog(s)

The earthquake catalog should be declustered and, ideally, classified according
to the source types of the earthquakes (i.e., subduction thrust, in-slab,
crustal, etc.).

The catalog(s) must be CSV files, with columns describing the fields and one row
for each earthquake.

### Temporal completeness

The catalog's temporal completeness can be specified in one of two ways:

1. **Simple date range**: Provide any two of `start_date`, `stop_date`, and
   `duration` in the YAML configuration. The third will be calculated. This
   assumes the catalog is complete above the minimum magnitude for the entire
   time period.

2. **Completeness table**: Provide a `completeness_table` as a list of
   `[year, magnitude]` pairs. Each pair specifies that the catalog is complete
   above that magnitude from that year onward. This allows Hamlet to account
   for varying completeness across the magnitude range, giving more accurate
   evaluations particularly for lower magnitudes with shorter complete periods.

### Separate catalogs for separate evaluations

If you are interested in running Hamlet separately for different seismic source
types, then make separate catalogs for the different earthquake categories,
i.e. make CSV files with only crustal events, only subduction megathrust
events, only slab events, etc.

Alternately if the different branches in the model's logic tree would for some
reason correspond to different subsets of the earthquake catalog, the catalog
should be split into separate files for each subset.


## Make the YAML configuration file(s)

See the [YAML configuration file documentation](yaml_config_file.html) for more information.

## Run Hamlet

Once the model, seismic catalog(s) and YAML configuration file(s) are prepared
(and Hamlet has been installed), Hamlet can be run like this:

```
hamlet test_ssm_crustal.yml
```

Additional command-line options:

- `hamlet --version` or `hamlet -v`: Print the version and exit.
- `hamlet config.yml --pdb`: Drop into the Python debugger on exception.
