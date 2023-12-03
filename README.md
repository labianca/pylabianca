# pylabianca
Python tools for spike analysis.

pylabianca:
* allows to read, analyse spike rate and statistically compare conditions in just a few steps
* follows the convenient API of mne-python
* provides two straightforward objects for storing spiking data: `Spikes` and `SpikeEpochs`
* allow storing trial-level metadata in these object (just like mne-python) and selecting trials based on these metadata
* returns xarrays (arrays with labeled dimensions and coordinates) as output from operations like cross-correlation, spiking rate calculation or decoding analysis
* these xarrays inherit all the trial-level metadata and can be visualised splitting by conditions using `pylabianca.viz.plot_shaded` or native xarray plotting
* the xarrays can be statistically tested with cluster based permutation test comparing condition metadata

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
