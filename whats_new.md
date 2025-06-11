# What's new

## DEV (upcoming version 0.4)

* DEV: Set up automated testing on CircleCI and code coverage tracking with codecov.com

<br/>

* API: new module `pylabianca.analysis` has been created. `pylabianca.utils.spike_centered_windows()`, `pylabianca.utils.shuffle_trials`, `pylabianca.utils.xarray_to_dict()` and `pylabianca.utils.dict_to_xarray()` have been moved there. These functions are still available in the `pylabianca.utils` namespace, but will now raise a deprecation warning when used from there. The functions will be removed from `pylabianca.utils` in the next major release (0.5).
* API: `pylabianca.viz.plot_waveform()` `y_bins` argument was renamed to `n_y_bins` to better reflect its meaning.
* API: `pylabianca.selectivity.compute_selectivity_continuous()` now returns an `xarray.Dataset` instead of dictionary of `xarray.DataArray` objects. This makes the output more convenient to work with (for example using `.sel()` to select elements of all items at the same time; or ability to use both `['key_name']` and `.key_name` to access items).
* API: removed, now unnecessary, `ignore_below` argument of `pylabianca.selectivity.depth_of_selectivity()` function. It was used to ignore firing rate values below a certain threshold, but the issue of very small non-zero values was previously fixed (numerical error likely stemming from the fact convolution used fft under the hood).
* API: removed `min_Hz` argument of `pylabianca.selectivity.compute_selectivity_continuous()`. It was used to ignore average firing rate values below a certain threshold, but such selection should be done by the user before calling the function.
* API: removed `pylabianca.utils.find_cells_by_cluster_id()` in favor of a more universal `pylabianca.utils.find_cells()`. Instead of doing `pln.utils.find_cells_by_cluster_id([254], channel='A15')` and then `pln.utils.find_cells_by_cluster_id([1854], channel='A23')` one can use `pln.utils.find_cells(cluster=[254, 1854], channel=['A15', 'A23'])`.

<br/>

* ENH: `Spikes` and `SpikeEpochs` can now be saved to FieldTrip data format. To maintain the input-output roundtrip (data saved and then read are identical) additional non-standard fields are added to the file when `.metadata` or `.cellinfo` are used. These additional fields should not conflict with using the file in FieldTrip.
* ENH: `SpikeEpochs.spike_rate()` now has a `center_time` argument to allow firing rate time coordinate to be centered around zero (default is `center_time=False`).
* ENH: `Spikes.epoch()` with `backend='numba'` has been further sped up, it is now 30-40 times faster than numpy
* ENH: `pylabianca.analysis.spike_centered_windows()` (previously `pylabianca.utils.spike_centered_windows()`) has been sped up twofold.
* ENH: `pylabianca.analysis.xarray_to_dict()` has been sped up considerably. It relies on sessions being concatenated along the cell dimension (so each session being a contiguous block of cells).
* ENH: add `pylabianca.utils._inherit_from_xarray()` to allow inheriting metadata (cell or trial-level additional information) from xarray DataArray to new xarray DataArray.
* ENH: `pylabianca.analysis.dict_to_xarray()` now allows to pass dictionary of `xarray.Dataset` as input.
* ENH: `SpikeEpochs.to_raw()` now takes `format` keyword argument, to allow binning SpikeEpochs into `mne.EpochsArray` object (`format='mne'`). The default is `format='numpy'`, which returns two numpy arrays (time and data array) as previously.
* ENH: added `pylabianca.selectivity.compute_percent_selective()` - function to use on the results of `pylabianca.selectivity.compute_selectivity_continuous()` to calculate the percentage of selective cells. Allows to split the calculations according to `groupby` argument value, specify selectivity threshold (single value or percentile of the permutation distribution). It computes the percentage on the permutation distribution (if provided) and uses it to calculate p value for the actual percentage selectivity. Can be used with time-resolved selectivity - the obtained time-resolved percentages for the actual data and the permutation distribution can be passed to a cluster-based permutation test.
* ENH: added `pylabianca.selectivity.threshold_selectivity()` to transform selectivity statistics xarray into binary selective / non-selective mask based on a given threshold (single value or percentile of the permutation distribution).
* ENH: added `pylabianca.analysis.aggregate()` to aggregate firing rate data. The aggregation is done by averaging the firing rate data over the trials dimension with optional grouping by one or more trial coordinates (conditions). The firing rate data can be optionally z-scored per-cell before aggregation. The function returns an xarray DataArray with aggregated firing rate data and accepts xarray DataArray or dictionary of xarray DataArray as input.
* ENH: added `pylabianca.analysis.zscore_xarray()` to z-score firing rate data in xarray DataArray. The z-scoring is done separately for each level of coordinate specified in `groupby` argument. Additional argument `baseline` allows to z-score the data using a baseline period (specified as a time range tuple or a separate xarray DataArray).
* ENH: added `pylabianca.selectivity.compute_selectivity_multisession()` to compute selectivity on a multisession dictionary (session name -> xarray). The output is an xarray.Dataset with concatenated session selectivity results (the order of the sessions in the output xarray.Dataset is the same as in the input dictionary).
* ENH: added `pylabianca.stats.find_percentile_threshold()` used to calculate significance threshold for given statistic based on percentile of the permutation distribution.
* ENH: `pylabianca.selectivity.compute_selectivity_continuous()` now can be also run with `n_perm=0`, returning only the selectivity statistics, without the permutation distribution or permutation-based threshold. Also `pylabianca.stats.permutation_test()` can now be run with `n_perm=0`, returning only the statistic values without the permutation distribution or permutation-based threshold.
* ENH: allow passing colors by name to `colors` in `pylabianca.viz.plot_shaded()`.

<br/>

* FIX: `pylabianca.analysis.xarray_to_dict()` used xarray `.groupby(session_coord)` to iterate over concatenated xarray and split it into dictionary of session name -> session xarray mappings. This had the unfortunate consequence of changing the order of sessions in the dictionary, if session order was not alphabetical in the concatenated xarray. Now `pylabianca.analysis.xarray_to_dict()` does not use `.groupby()` and preserves the order of sessions in the dictionary.
* FIX: make `pylabianca.analysis.xarray_to_dict()` work also on arrays without cell x trial multi-dim coords (e.g. `('cell', 'trial')`), which are common after concatenating multiple sessions.
* FIX: saving pylabianca created xarrays like firing rate to NetCDF now works without the need to clear attributes (previously a dictionary of coord units was stored in the attributes, which caused an error when writing the file).
* FIX: `SpikeEpochs.n_spikes(per_epoch=True)` used spike rate calculation to count spikes in each epoch. This was unnecessary (and possibly slow) and in rare cases could lead to wrong results (probably numerical error when multiplying spike rate by window duration and immediately turning to int, without rounding). Now `SpikeEpochs.n_spikes(per_epoch=True)` counts spikes directly using `pylabianca.utils._get_trial_boundaries`.
* FIX: mne compatibility - use the `copy=False` argument in `.get_data()` method (introduced in newer mne versions)
* FIX: made `Spikes.epoch()` raise a more informative error when no `event_id` values were found in the provided `events` array. When some of the `event_id` values are missing, a warning is raised and the function proceeds with the available values.
* FIX: fixed error when passing a single integer to `event_id` in `Spikes.epoch()`.
* FIX: fixed error when trying to epoch cells with no spikes
* FIX: fixed error when using `Spikes.epoch()` with `backend='numba'` when some epoch limits did not contain spikes. Each such epoch would still receive one spike, but with correctly calculated time (out of epoch range). This way this bug did not affect further spike rate analysis.
* FIX: small fixes to `pylabianca.postproc.mark_duplicates()` - do not error when there are channels without any spikes.
* FIX: dataframe returned by `pylabianca.selectivity.cluster_based_selectivity()` had two unused columns (`'pev'` and `'peak_pev'`), where correct names should have been `'PEV'` and `'peak_PEV'`. Now corrected.
* FIX: make `pylabianca.selectivity.assess_selectivity()` work when empty DataFrame is passed (no clusters found in `pylabianca.selectivity.cluster_based_selectivity()`).
* FIX: `pylabianca.viz.plot_shaded()` auto-inferring of dimension to reduce (average) now correctly ignores `groupby` argument (if identical to one of the dimensions).
* FIX: when using groupby in `pylabianca.viz.plot_shaded()` the axis title would often display the label of last condition plotted. This is now fixed, the groupby condition is not shown in the title.
* FIX: `pylabianca.viz.plot_shaded()` now produces a clearer error when too many dimensional DataArray is used.
* FIX: `pylabianca.utils._handle_cell_names()` (used internally in a few places) now works with NumPy >= 2.0.
* FIX: `pylabianca.viz.plot_waveform()` (as well as `.plot_waveform()` methods of `Spikes` and `SpikeEpochs`) now produces a clearer error when no waveforms are present in the data.
* FIX: fixed `pval_text=False` still giving p value text (but without text boxes) in `pylabianca.viz.add_highlights()`
* FIX: using `pylabianca.stats.cluster_based_test()` would often give (1, n) shaped clusters and passing these to `pylabianca.viz.add_highlighs()` lead to errors. This has now been fixed - such clusters would be ravel()'ed into 1d representation.

<br/>

* DOC: add docstring to `pylabianca.selectivity.assess_selectivity()`
* DOC: improve data format docs about FieldTrip data format.

<br/><br/>

## Version 0.3
* much increased code coverage of automated tests

<br/>

* API: `per_trial=False` option was removed from `.apply()` method of `SpikeEpochs` - it didn't seem to be useful and its behavior was not well defined. If you need to apply a function to each trial separately, you can still use `.apply()`.
* API: remove unnecessary `spk` argument from `pylabianca.selectivity.cluster_based_selectivity()`. It was used to calculate spike rate and various selectivity indices from cluster-test defined window. Now the function calculates this measures from the provided firing rate xarray.
* API: removed `feature_selection` argument from `pylabianca.decoding.run_decoding()` and `pylabianca.decoding.run_decoding_array()` functions. It was not very useful and it is better to place feature selection as part of the sklearn pipeline passed to these functions.

<br/>

* ENH: around 10-fold speed up to `Spikes.epoch()` (20-fold for thousands of spikes and epoching events)
* ENH: further speed up to `Spikes.epoch()` (around 5 - 13-fold) is now also possible by using `backend='numba'` (if numba is installed)
* ENH: added `n_jobs` argument to `pylabianca.selectivity.cluster_based_selectivity()` to allow for parallel processing of cells
* ENH: allow for different percentile level in `pylabianca.stats.cluster_based_test_from_permutations()` using `percentile` argument.

* ENH: `.plot_waveform()` method of `Spikes` and `SpikeEpochs` now allows to control the colormap to plot the waveform density with (`cmap` argument) and the number of y axis bins (`y_bins` argument)
* ENH: added `colors` argument for explicit color control in `pylabianca.viz.plot_raster`
* ENH: `pylabianca.viz.plot_raster` now creates legend for condition colors. This is the default behavior, but `legend` argument allows to control this, and `legend_kwargs` allows for  passing additional arguments to the legend
* ENH: added an experimental datashader backend to `.plot_waveform()` method of `Spikes` and `SpikeEpochs` (`backend='datashader'`).
* ENH: allow to pass arguments to eventplot via `eventplot_kwargs` from `pylabianca.viz.plot_raster()` and `pylabianca.viz.plot_spikes()`

* ENH: added `pylabianca.utils.cellinfo_from_xarray()` function to extract/reconstruct cellinfo dataframe from xarray DataArray coordinates.
* ENH: `pylabianca.utils.xr_find_nested_dims()` now returns "nested" xarray coordinates also for coordinate tuples (for example `('cell', 'trial')` - which happens often after concatenating multiple xarray sessions)
* ENH: added `copy_cellinfo` argument to `pylabianca.selectivity.cluster_based_selectivity()`. It allows to select which cellinfo columns are copied to the selectivity dataframe.
* ENH: expose `.to_spiketools()` as `SpikeEpochs` method (previously it was only available as a function in `pylabianca.io` module)
* ENH: added `pylabianca.utils.dict_to_xarray()` function to convert dictionary of xarrays (multiple sessions / subjects) to one concatenated xarray DataArray
* EHN: added `pylabianca.utils.xarray_to_dict()` function to convert xarray DataArray to dictionary of xarrays (useful when splitting xarray DataArray to multiple sessions / subjects)
* ENH: added `pylabianca.utils.assign_session_coord()` function to assign session / subject coordinate to xarray DataArray (useful when concatenating multiple sessions / subjects- using `pylabianca.utils.dict_to_xarray()`)

* ENH: allow to select trials with boolean mask for `SpikeEpochs` objects (e.g. `spk_epochs[np.array([True, False, True])]` or `spk_epochs[my_mask]` where `my_mask` is a boolean array of length `len(spk_epochs)`)
* ENH: `Spikes` `.sort()` method now exposes `inplace` argument to allow for sorting on a copy of the object (this can be also easily done by using `spk.copy().sort()`)

* ENH: add `use_usenegative` argument to `pylabianca.io.read_osort()`. It allows to read only clusters / units indicated by `usenegative` field in the OSort output file. This field is used when exporting data from OSort in a lazy way (not removing unused clusters from the data file, but instead just adding / modifying `usenegative` field).
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
