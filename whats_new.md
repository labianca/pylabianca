# What's new

## Version 0.2

* ENH: added `pylabianca.io.from_spiketools()` function to convert data from list of arrays format used by spiketools to pylabianca `SpikeEpochs`
* ENH: added `pylabianca.io.to_spiketools()` function to convert data from pylabianca `SpikeEpochs` to list of arrays format used by spiketools
* ENH: added `pylabianca.io.read_analog_plexon_nex()` function to read analog (continuous) data from Plexon NEX files. The continuous data are stored in `mne.io.RawArray` object.
* ENH: added `.apply()` method to `SpikeEpochs` allowing to run arbitrary functions on the spike data. At the moment the function has to take one trial (or all trials) and return a single value.
* ENH: improved `pylabianca.utils.spike_centered_windows()` to handle xarray DataArrays and mne.Epochs objects. Also, the returned windows xarray now inherits metadata from SpikeEpochs object (or from mne.Epochs object if
the SpikeEpochs object does not contain metadata).
* ENH: added option to store original timestamps when epoching (`keep_timestamps` argument). These timestamps are kept in sync with event-centered spike times through all further operations like condition selection, cropping, etc.
* ENH: to increase compatibility with MNE-Python `len(SpikeEpochs)` returns the number of trials now. To get the number of units use `SpikeEpochs.n_units()`
* ENH: added `pylabianca.utils.shuffle_trials()` function to shuffle trials in `SpikeEpochs` object

* DOC: added example of working with pylabianca together with spiketools: [notebook](doc/working_with_spiketools.ipynb)
* DOC: added example of spike-field analysis combining pylabianca and MNE-Python: [notebook](doc/spike-triggered_analysis.ipynb)


* FIX: removed incorrect condition label [FIXME: add more info] `pylabianca.viz.plot_shaded()`
* FIX: `.plot_isi()` `Spikes` method was not committed in 0.1 version, now added. It is just a wrapper around `pylabianca.viz.plot_isi()`
* FIX: typo in `pylabianca.io.read_fieldtrip` (was `read_filedtrip` before)
* FIX: when `.metadata` attribute of `SpikeEpochs` is set the metadata is tested to be a DataFrame with relevant number of rows. Also, row indices are reset to match trial indices.
