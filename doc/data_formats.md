# Spike data formats

Pylabianca natively reads Osort, Combinato and FieldTrip data formats.
Other formats can be easily read via python [neo](https://neo.readthedocs.io/en/latest/) package. An example of reading spike data from Plexon `.nex` file (NeuroExplorer) using neo can be found [here](doc/spike-triggered_analysis.ipynb).
Support for other formats will be added in future. If there is a file format you would like to read directly to pylabianca, without using neo - drop us an issue on GitHub!

## FieldTrip data format
The format used by fieldtrip is described in more detail in [the FieldTrip spike analysis tutorial](https://www.fieldtriptoolbox.org/tutorial/spike/).
This format uses matlab files (`*.mat`).

### raw spikes
FieldTrip format for continuous ("raw") spikes. This format is used to represent spike times prior to epoching with respect to events of interest. Spike times are stored in timestamps.
The data variable contains a structure with the following fields:
```
        label: {1×21 cell}
    timestamp: {1×21 cell}
     waveform: {1×21 cell}
         unit: {1×21 cell}
          hdr: [1×1 struct]
       dimord: '{chan}_lead_time_spike'
```
* `label` - just like in LFP (channel or cell names)
* `timestamp` - `1 x n_neurons` cell array where each cell contains `1 x n_spikes` vector of timesamples when spike occured
* `waveform` - `1 x n_neurons` cell array where each cell contains `n_leads x n_samples x n_spikes` matrix of spike waveforms (`n_samples` is usually low, for the example fieldtrip data it is 32, but it depends on the sampling rate). A cell of the array can be empty - this means that there are no waveforms for given neuron.
* `unit` - 1 x n_neurons cell array with 1 x n_spikes vector of what seems to be unit cluster ID's (in the example data it is all nan)
* `hdr` - file header information

#### reading / writing
This format can be read by `pylabianca.io.read_fieldtrip(path_to_file, kind='raw')`.
To save to this file format use `.to_fieldtrip()` method of `pylabianca.Spikes` object. When saving to fieldtrip raw spikes format the following additional fields may be stored in the file. Information stored in these fields does not have an equivalent in standard fieldtrip format:
* `cellinfo` - a structure containing field corresponding to columns of `Spikes.cellinfo` pandas DataFrame. Each field stores an array / cell array of `n_neurons` length. Added only if `Spikes.cellinfo` is not `None`.
* `waveform_time` - `n_samples` array of time labels corresponding to waveforms time dimension. Added only if `Spikes.waveform_time` attribute is not `None`.


### spikeTrials
FieldTrip format for epoched spiking data. We now additionally have the following fields:
* `time` - spike times in seconds with respect to trigger onset
* `trial` for each spike - information about the trial in which it fires
* `trialtime`:  `n_trials x 2` (time in seconds of the trial beginning and end - for each trial)
* `trialinfo` - optional field with `n_trials x N` array. Contains trial-level additional information (can store data such as participant response, correctness, reaction time, trial condition, etc.)

#### reading / writing
This format can be read by `pylabianca.io.read_fieldtrip(path_to_file, kind='trials')`.
To save to this file format use `.to_fieldtrip()` method of `pylabianca.SpikeEpochs` object. When saving to fieldtrip spikeTrials format the following additional fields may be stored in the file. Information stored in these fields does not have an equivalent in standard fieldtrip format:
* `trialinfo_columns` - column names for `trialinfo`. Added only if `SpikeEpochs.metadata` is not `None` (in which case `trialinfo` field is added).
* `cellinfo` - a structure containing field corresponding to columns of `SpikeEpochs.cellinfo` pandas DataFrame. Each field stores an array / cell array of `n_neurons` length. Added only if `SpikeEpochs.cellinfo` is not `None`.
* `waveform_time` - `n_samples` array of time labels corresponding to waveforms time dimension. Added only if `SpikeEpochs.waveform_time` attribute is not `None`.

## Osort output data
Format used as output by the Osort sorter.
This format uses matlab files (`*.mat`).

### standard format
The data variable is a structure with the following fields:
* `assignedNegative` - a vector of cluster id per each spike
* `newSpikesNegative` - waveforms, one per spike
* `newTimestampsNegative` - timestamps for all the spikes
* allSpikesCorrFree - also waveforms, but different from newSpikesNegative (FIX - more info needed here)
* scalingFactor - scaling factor for the waveforms (?), currently not used because it frequently is NaN and the read scaling factor has to be read from Neuralynx file header.  (FIX - more info needed here)

This format uses one file per lead (micro-electrode), so a path to folder (not file) is needed to read multiple channels.
This format can be read by `pylabianca.io.read_osort(path_to_folder, format='standard')`.
This function can be used to read only some of the channels, reading vs not reading waveforms etc. - for more details see the docstring.

### "mm" format
Slightly modified and cleaned up variant of the Osort output format.
The data variable is a structure with the following fields:
* `cluster_id` - vector of cluster ids, one per cell (not one per spike as in "standard" format)
* `timestamp` - cell array, where each cell contains vector of timestamps for respective neuron (so `data.timestamp{1}` contains timestamps for the unit formed by the first cluster in `data.cluster_id`)
* `waveform` - cell array, where each cell contains `n_spikes x n_samples` matrix of waveforms for respective neuron
* `alignment` - cell array of alignment information (the aligment - negative, positive or mixed for the chosen unit)
* `threshold` - vector of spike SD thresholds used for spike detection
* `channel` - cell array of channel names (one channel name per

This format uses only one file per sorting (not one per lead/micro-channel as in Osort "standard" format).
This format can be read by `pylabianca.io.read_osort(path_to_file, format='mm')`.
This function can be used to read only some of the channels, reading vs not reading waveforms etc. - for more details see the docstring.

## Combinato output data
Standard combinato multi-folder data.

This format can be read by `pylabianca.io.read_combinato(path, alignment='both')`.
