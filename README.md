# pylabianca: Python tools for spike analysis

[![labianca](https://circleci.com/gh/labianca/pylabianca.svg?style=svg)](https://app.circleci.com/pipelines/github/labianca/pylabianca)
[![codecov](https://codecov.io/gh/labianca/pylabianca/graph/badge.svg?token=HQ7KN5FWL5)](https://codecov.io/gh/labianca/pylabianca)

pylabianca offers a simple and efficient way to read, analyze, and statistically compare spike data in just a few steps. Key features include:

* A familiar API inspired by [mne-python](https://mne.tools/stable/index.html), ensuring ease of use for experienced users.
* Two intuitive data structures - `Spikes` and `SpikeEpochs` - for organizing and storing spike data.
* Integrated support for storing trial-level metadata, enabling easy trial selection based on conditions, similar to mne-python.
* Outputs in the form of [xarray](https://docs.xarray.dev/en/stable/) DataArrays, which come with labeled dimensions and coordinates.
* Seamless metadata inheritance in xarrays, allowing for visualizations by condition using `pylabianca.viz.plot_shaded` or native xarray plotting functions.
* Built-in support for statistical testing via cluster-based permutation tests, facilitating comparisons between different conditions based on trial metadata.

## installation
`pylabianca` can be installed using `pip`:
```
pip install pylabianca
```
To get most up-to-date version you can also install directly from github:
```
pip install git+https://github.com/labianca/pylabianca
```

## what's new?
See [whats_new.md](whats_new.md) for documentation of recent changes in pylabianca.

## docs
Online docs are currently under construction.

Below you can find jupyter notebook examples showcasing `pylabianca` features.
* [introductory notebook](doc/intro_overview.ipynb) - a general overview using human intracranial spike data (sorted with Osort).
* [FiedTrip data example notebook](doc/fieldtrip_example.ipynb) - another broad overview using fieldtrip sample spike data from non-human primates.
* [decoding example](doc/decoding_example.ipynb) - overview of decoding with pylabianca
* [spike-triggered LFP analysis](doc/spike-triggered_analysis.ipynb) - use pylabianca and [`MNE-Python`](https://github.com/mne-tools/mne-python) to perform spike-triggered analysis of LFP
* [working with spiketools](doc/working_with_spiketools.ipynb) - example of how [`spiketools`](https://github.com/spiketools/spiketools) and pylabianca can be used together

To better understand the data formats read natively by pylabianca (and how to read other formats) see [data formats page](doc/data_formats.md).

### sample data

You can get example human data that are used in the examples [here](https://www.dropbox.com/scl/fo/wevgovmxv8qrl52w12b6z/h?rlkey=1je64v2h1h6zyqhzmhiykpqqu&dl=0).  
The preprocessed FieldTrip data used in the examples are available [here](https://www.dropbox.com/scl/fo/i6q4e0ix805dds92jibmw/h?rlkey=cfdm1730qubqwb64zj1j02tvt&dl=0).
