# What's new

## DEV (upcoming version 0.3)
* API: `per_trial=False` option was remove from `.apply()` method of `SpikeEpochs` - it didn't seem to be useful and its behavior was not well defined. If you need to apply a function to each trial separately, you can still use `.apply()`.


* ENH: expose `.to_spiketools()` as `SpikeEpochs` method (previously it was only available as a function in `pylabianca.io` module)
* ENH: allow to select trials with boolean mask for `SpikeEpochs` objects (e.g. `spk_epochs[np.array([True, False, True])]` or `spk_epochs[my_mask]` where `my_mask` is a boolean array of length `len(spk_epochs)`)
* ENH: `Spikes` `.sort()` method now exposes `inplace` argument to allow for sorting on a copy of the object (this can be also easily done by using
`spk.copy().sort()`)


* DOC: added dosctring to `pylabianca.stats.permutation_test()`


* FIX: allow to `.drop_cells()` using cell names, not only indices
* FIX: `Spikes` `.sort()` method now raises error when using `Spikes` with empty `.cellinfo` attribute or when the attribute does not contain a pandas DataFrame.
* FIX: make sure `.n_trials` `SpikeEpochs` attribute is int


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
* ENH: speed up auto- and cross-correlation for `Spikes` with more loopy approach, previous `spk.to_epochs().xcorr()` approach was insanely slow for large datasets. In general, for data with lots of spikes it is more efficient to use a combination of smart looping + `np.histogram()` than to use numpy for all the calculations.
* ENH: added `.xcorr()` method to `Spikes` objects
* ENH: added a very fast numba implementation of auto- and cross-correlation for `Spikes` objects.
* ENH: A new argument `backend` was added to `.xcorr()` method of `Spikes`. The default value is `backend='auto'`, which automatically selects the backend (if numba is available and there are many spikes in the data, numba is used). Other options are `backend='numpy'` and `backend='numba'`.


* DOC: added example of working with pylabianca together with spiketools: [notebook](doc/working_with_spiketools.ipynb)
* DOC: added example of spike-field analysis combining pylabianca and MNE-Python: [notebook](doc/spike-triggered_analysis.ipynb)


* FIX: removed incorrect condition label [FIXME: add more info] `pylabianca.viz.plot_shaded()`
* FIX: `.plot_isi()` `Spikes` method was not committed in 0.1 version, now added. It is just a wrapper around `pylabianca.viz.plot_isi()`
* FIX: typo in `pylabianca.io.read_fieldtrip` (was `read_filedtrip` before)
* FIX: when `.metadata` attribute of `SpikeEpochs` is set the metadata is tested to be a DataFrame with relevant number of rows. Also, row indices are reset to match trial indices.
