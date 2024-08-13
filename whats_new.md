# What's new

## DEV (upcoming version 0.3)
* much increased code coverage of automated tests

<br/>

* API: `per_trial=False` option was removed from `.apply()` method of `SpikeEpochs` - it didn't seem to be useful and its behavior was not well defined. If you need to apply a function to each trial separately, you can still use `.apply()`.
* API: remove unnecessary `spk` argument from `pylabianca.selectivity.cluster_based_selectivity()`. It was used to calculate spike rate and various selectivity indices from cluster-test defined window. Now the function calculates this measures from the provided firing rate xarray.
* API: removed `feature_selection` argument from `pylabianca.decoding.run_decoding()` and `pylabianca.decoding.run_decoding_array()` functions. It was not very useful and it is better to place feature selection as part of the sklearn pipeline passed to these functions.

<br/>

* ENH: implement fast ZETA test for comparing trials from two up to N conditions for each cell. The test is implemented in `pylabianca.selectivity.zeta_test()`. ZETA is a non-parametric test that compares the cumulative distributions of firing rates between conditions (see https://elifesciences.org/articles/71969). The current implementation is based on the zetapy package (https://pypi.org/project/zetapy/), but is much faster and compatible with pylabianca SpikeEpochs objects. The speedup with respect to zetapy depends on the computation backend used (``backend`` argument): the default numpy implementation is around 5 - 10 faster, while the numba implementation is around 20 - 40 times faster.
* ENH: around 10-fold speed up to `Spikes.epoch()` (20-fold for thousands of spikes and epoching events)
* ENH: further speed up to `Spikes.epoch()` (around 5 - 13-fold) is now also possible by using `backend='numba'` (if numba is installed)
* ENH: added `n_jobs` argument to `pylabianca.selectivity.cluster_based_selectivity()` to allow for parallel processing of cells
ENH: allow for different percentile level in `pylabianca.stats.cluster_based_test_from_permutations()` using `percentile` argument.

* ENH: `.plot_waveform()` method of `Spikes` and `SpikeEpochs` now allows to control the colormap to plot the waveform density with (`cmap` argument) and the number of y axis bins (`y_bins` argument)
* ENH: added `colors` argument for explicit color control in `pylabianca.viz.plot_raster`
* ENH: `pylabianca.viz.plot_raster` now creates legend for condition colors. This is the default behavior, but `legend` argument allows to control this, and `legend_kwargs` allows for  passing additional arguments to the legend
* ENH: added an experimental datashader backend to `.plot_waveform()` method of `Spikes` and `SpikeEpochs` (`backend='datashader'`).
* ENH: allow to pass arguments to eventplot via `eventplot_kwargs` from `pylabianca.viz.plot_raster()` and `pylabianca.viz.plot_spikes()`

* ENH: added `pylabianca.utils.cellinfo_from_xarray()` function to extract/reconstruct cellinfo dataframe from xarray DataArray coordinates.
* ENH: `pylabianca.utils.xr_find_nested_dims()` now returns "nested" xarray coordinates also for coordinate tuplples (for example `('cell', 'trial')` - which happens often after concatenating multiple xarray sessions)
* ENH: added `copy_cellinfo` argument to `pylabianca.selectivity.cluster_based_selectivity()`. It allows to select which cellinfo columns are copied to the selectivity dataframe.
* ENH: expose `.to_spiketools()` as `SpikeEpochs` method (previously it was only available as a function in `pylabianca.io` module)
* ENH: added `pylabianca.utils.dict_to_xarray()` function to convert dictionary of xarrays (multiple sessions / subjects) to one concatenated xarray DataArray
* ENH: added `pylabianca.utils.assign_session_coord()` function to assign session / subject coordinate to xarray DataArray (useful when concatenating multiple sessions / subjects- using `pylabianca.utils.dict_to_xarray()`)

* ENH: allow to select trials with boolean mask for `SpikeEpochs` objects (e.g. `spk_epochs[np.array([True, False, True])]` or `spk_epochs[my_mask]` where `my_mask` is a boolean array of length `len(spk_epochs)`)
* ENH: `Spikes` `.sort()` method now exposes `inplace` argument to allow for sorting on a copy of the object (this can be also easily done by using `spk.copy().sort()`)

* ENH: better error message when the format passed to `pylabianca.io.read_osort()` does not match the data
* ENH: added better input validation to `SpikeEpochs` to avoid silly errors
* ENH: added better input validation to `Spikes` to avoid silly errors
* ENH: when adding or modifying `.cellinfo` it is now verified to have correct format and length
* ENH: limited dependency on `matplotlib`, `sklearn`, `mne` and `borsar` - they are now used only in functions that require them

<br/>

* DOC: added docstring to `pylabianca.stats.permutation_test()`
* DOC: add missing entries to docstring of `pylabianca.selectivity.cluster_based_selectivity()`
* DOC: improved the FieldTrip data example in the documentation: [notebook](doc/fieldtrip_example.ipynb)

<br/>

* FIX: correct how permutations are handled in `pylabianca.decoding.resample_decoding()`. Previously each resample (random trial matching between sessions to create a pseudo-population) within one permutation step used different trial-target matchings (although the same permutation vector was passed to each resample). Because all resamples are averaged, to get a better estimate of the true effect, this lead to overly optimistic p-values (different trial-target matchings averaged in each permutation, but the same matching averaged in non-permuted data). Now the permutation of the target vector is done per-session, before pseudo-population creation / resampling, which fixes the issue.
* FIX: numerical errors in `.spike_density()` method of `SpikeEpochs`. The scipy's convolution used in this function automatically picks FFT-based convolution for larger arrays, which leads to close-to-zero noise (around 1e-15) where it should be zero. We now use overlap-add convolution (`scipy.signal.oaconvolve`) which has very similar speed and less severe close-to-zero numerical errors. Additionally, values below `1e-14` are now set to zero after convolution.
* FIX: better handle cases where some cells have 0 spikes after epoching. Previously, this lead to errors when constructing epochs or calculating firing rate, now it is handled gracefully (changes to `pylabianca.utils._get_trial_boundaries()`, `pylabianca.utils.is_list_of_non_negative_integer_arrays()` and `pylabianca.spike_distance._xcorr_hist_trials` accessed through `SpikeEpochs.xcorr()`).
* FIX: better handle cases when there is not enough data (time range) for spike rate window length (`SpikeEpochs.spike_rate()`) or convolution kernel length (`SpikeEpochs.spike_density()`)
* FIX: allow to `.drop_cells()` using cell names, not only indices
* FIX: `Spikes` `.sort()` method now raises error when using `Spikes` with empty `.cellinfo` attribute or when the attribute does not contain a pandas DataFrame.
* FIX: make sure `.n_trials` `SpikeEpochs` attribute is int
* FIX: fix calculation of the time vector in spike rate calculation - in some cases the firing rate array had one less element than the time vector, because they were calculated independently (this was due to floating point arithmetic) and lead to an error when constructing xarray DataArray.
* FIX: avoid error when one-tail test (ANOVA) is used in `pylabianca.selectivity.compute_selectivity_continuous()` - previously the function assumed that always two-tail thresholds are returned
* FIX: make sure cellinfo columns inherited by firing rate xarray survive through `pylabianca.selectivity.compute_selectivity_continuous()`
* FIX: fix error when no spikes were present in fixed time window in `pylabianca.spike_rate._compute_spike_rate_fixed()` used when `step=False` in `.spike_rate()` method of `SpikeEpochs`
* FIX: fix bug in `pylabianca.io.read_events_neuralynx()`: if no reference file was given to get the first timestamp from - an error was thrown. Now we fill the start column (time in seconds from recording start) with NaN
* FIX: plotting only one line passing one color to `pylabianca.viz.plot_shaded()` or `pylabianca.viz.plot_xarray_shaded()` now works correctly (previously it resulted in an error)
<br/><br/>

## Version 0.2

* ENH: added `pylabianca.io.from_spiketools()` function to convert data from list of arrays format used by spiketools to pylabianca `SpikeEpochs`
* ENH: added `pylabianca.io.to_spiketools()` function to convert data from pylabianca `SpikeEpochs` to list of arrays format used by spiketools
* ENH: added `pylabianca.io.read_signal_plexon_nex()` function to read continuous data from Plexon NEX files. The continuous data are stored in `mne.io.RawArray` object.
* ENH: added `.apply()` method to `SpikeEpochs` allowing to run arbitrary functions on the spike data. At the moment the function has to take one trial (or all trials) and return a single value.
* ENH: improved `pylabianca.utils.spike_centered_windows()` to handle xarray DataArrays and mne.Epochs objects. Also, the returned windows xarray now inherits metadata from SpikeEpochs object (or from mne.Epochs object if the SpikeEpochs object does not contain metadata).
* ENH: added option to store original timestamps when epoching (`keep_timestamps` argument). These timestamps are kept in sync with event-centered spike times through all further operations like condition selection, cropping, etc.
* ENH: to increase compatibility with MNE-Python `len(SpikeEpochs)` returns the number of trials now. To get the number of units use `SpikeEpochs.n_units()`
* ENH: added `pylabianca.utils.shuffle_trials()` function to shuffle trials in `SpikeEpochs` object
* ENH: speed up auto- and cross-correlation for `Spikes` with more loopy approach, previous `spk.to_epochs().xcorr()` approach was insanely slow for large datasets. In general, for data with lots of spikes it is more efficient to use a combination of smart looping + `np.histogram()` than to use numpy for all the calculations.
* ENH: added `.xcorr()` method to `Spikes` objects
* ENH: added a very fast numba implementation of auto- and cross-correlation for `Spikes` objects.
* ENH: A new argument `backend` was added to `.xcorr()` method of `Spikes`. The default value is `backend='auto'`, which automatically selects the backend (if numba is available and there are many spikes in the data, numba is used). Other options are `backend='numpy'` and `backend='numba'`.

<br/>

* DOC: added example of working with pylabianca together with spiketools: [notebook](doc/working_with_spiketools.ipynb)
* DOC: added example of spike-field analysis combining pylabianca and MNE-Python: [notebook](doc/spike-triggered_analysis.ipynb)

<br/>

* FIX: removed incorrect condition label when using `pylabianca.viz.plot_shaded()` with `groupby` argument. Previously the last condition label was used for the figure, although one line per condition was shown.
* FIX: `.plot_isi()` `Spikes` method was not committed in 0.1 version, now added. It is just a wrapper around `pylabianca.viz.plot_isi()`
* FIX: typo in `pylabianca.io.read_fieldtrip` (was `read_filedtrip` before)
* FIX: when `.metadata` attribute of `SpikeEpochs` is set the metadata is tested to be a DataFrame with relevant number of rows. Also, row indices are reset to match trial indices.
