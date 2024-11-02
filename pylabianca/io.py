import os
import os.path as op
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd

from .spikes import SpikeEpochs, Spikes
from .utils import _deal_with_picks, _get_trial_boundaries


# CONSIDER - autodetect the correct var name
def read_fieldtrip(fname, data_name='spike', kind='raw', waveform=True):
    '''Read fieldtrip SpikeTrials format.

    Parameters
    ----------
    fname : str | pathlib.Path
        Path to the file to read.
    data_name : str
        The name of the saved variable - this can be arbitrary so it is
        necessary to specify. ``'spike'`` by default.
    kind : str
        Data format to read. Can be:
        * ``'raw'`` for FieldTrip raw spikes format
        * ``'trials'`` or ``'epochs'`` for FieldTrip SpikeTrials format
    waveform : bool
        Whether to read waveforms. Defaults to ``True``.

    Returns
    -------
    spk : SpikeEpochs
        SpikeEpochs object.
    '''
    from scipy.io import loadmat

    accept_kind = ['epochs', 'trials', 'raw']
    if kind not in accept_kind:
        msg = (f'`kind` has to be one of {accept_kind}, got {kind}.')
        raise ValueError(msg)

    data = loadmat(fname, squeeze_me=True, variable_names=data_name)[data_name]

    cell_names = data['label'].item()

    if waveform:
        waveform, waveform_time = _get_waveforms(data)
    else:
        waveform, waveform_time = None, None

    trial_info = _get_ft_trial_info(data)
    cellinfo = _get_ft_cellinfo(data)

    if kind in ['trials', 'epochs']:
        spk = _read_ft_spikes_tri(data, cell_names, trial_info, cellinfo,
                                  waveform, waveform_time)
    elif kind == 'raw':
        spk = _read_ft_spikes_raw(data, cell_names, cellinfo,
                                  waveform, waveform_time)

    spk.filename = fname
    return spk


def _read_ft_spikes_tri(data, cell_names, trial_info, cellinfo, waveform,
                        waveform_time):
    '''Read fieldtrip SpikeTrials format.

    Returns
    -------
    spk : SpikeEpochs
        SpikeEpochs object.
    '''

    time = data['time'].item()
    trial = data['trial'].item() - 1
    trial_time = data['trialtime'].item()

    if 'timestamp' in data.dtype.names:
        timestamps = data['timestamp'].item()
    else:
        timestamps = None

    msg = 'All trials have to be of the same length'
    assert (trial_time == trial_time[[0]]).all(), msg

    n_trials = trial_time.shape[0]
    time_limits = tuple(trial_time[0])

    # create SpikeEpochs
    spk = SpikeEpochs(time, trial, time_limits, n_trials=n_trials,
                      metadata=trial_info, cell_names=cell_names,
                      cellinfo=cellinfo, waveform=waveform,
                      waveform_time=waveform_time, timestamps=timestamps)

    return spk


def _read_ft_spikes_raw(data, cell_names, cellinfo, waveform,
                        waveform_time):
    '''Read raw spikes fieldtrip format.

    Returns
    -------
    spk : Spikes
        Spikes object.

    '''
    timestamps = data['timestamp'].item()
    timestamps = _check_timestamps(timestamps)
    sfreq = data['hdr'].item()['FileHeader'].item()['Frequency'].item()

    # create Spikes
    spk = Spikes(timestamps, sfreq, cell_names=cell_names,
                 cellinfo=cellinfo, waveform=waveform,
                 waveform_time=waveform_time)
    return spk


def _check_timestamps(timestamps):
    from numbers import Integral

    n_units = len(timestamps)
    for ix in range(n_units):
        if isinstance(timestamps[ix], Integral):
            timestamps[ix] = np.array([timestamps[ix]])

    return timestamps


def _get_waveforms(data):
    '''Get waveforms from a FieldTrip file.'''
    has_waveform = 'waveform' in data.dtype.names

    if not has_waveform:
        waveforms, waveform_time = None, None
        return waveforms, waveform_time

    has_waveform_time = 'waveform_time' in data.dtype.names
    if has_waveform_time:
        sfreq = None
        waveform_time = data['waveform_time'].item()
    else:
        sfreq = _get_ft_sfreq(data)
        waveform_time = None

    waveform_data = data['waveform'].item()
    waveforms = list()
    n_units = len(waveform_data)
    for ix in range(n_units):
        this_waveform = waveform_data[ix]

        if (isinstance(this_waveform, np.ndarray)
            and len(this_waveform) > 0):

            # we currently take only the first lead
            if this_waveform.ndim > 2:
                this_waveform = this_waveform[0]
            elif this_waveform.ndim == 1:
                # only one waveform, squeezed
                this_waveform = this_waveform[:, None]
            waveforms.append(this_waveform.T)
        else:
            waveforms.append(None)

    waveform_length = [x.shape[1] for x in waveforms if x is not None]
    same_lengths = all(waveform_length[0] == x
                       for x in waveform_length[1:])

    if same_lengths:
        # waveform time in ms
        if waveform_time is None and sfreq is not None:
            waveform_time = np.arange(waveform_length[0]) / (sfreq / 1000)
    else:
        warn('Not all waveforms have the same number of samples, '
             'waveforms are therefore ignored. Got the following waveform'
             f'lengths: {waveform_length}.')
        waveforms, waveform_time = None, None

    return waveforms, waveform_time


def _get_ft_sfreq(data):
    try:
        sfreq = data['hdr'].item()['FileHeader'].item()['Frequency'].item()
    except ValueError:
        # no field
        sfreq = None

    return sfreq


def _get_ft_trial_info(data):
    if 'trialinfo' in data.dtype.names:
        trial_info = data['trialinfo'].item()
        if 'trialinfo_columns' in data.dtype.names:
            columns = data['trialinfo_columns'].item()
            columns = [col.rstrip() for col in columns]
        else:
            columns=None
        trial_info = pd.DataFrame(trial_info, columns=columns)
        trial_info = trial_info.infer_objects()
    else:
        trial_info = None

    return trial_info


def _get_ft_cellinfo(data):
    if 'cellinfo' in data.dtype.names:
        data_dct = dict()
        fields = data['cellinfo'].item().dtype.names

        for fld in fields:
            data_extr = data['cellinfo'].item()[fld]
            try:
                data_extr = data_extr.item()
            except ValueError:
                pass
            data_dct[fld] = data_extr

        cellinfo = pd.DataFrame(data_dct)
    else:
        cellinfo = None

    return cellinfo


# - [ ] full_ft_compat (or sth similar) to only save trialinfo columns that
#       are compatible with fieldtrip?
# - [ ] consider saving sampling frequency if available:
#       data['hdr'].item()['FileHeader'].item()['Frequency'].item()
#       (what if waveform_time and timestamp sfreq are different ?
#        first check if waveform_time is present ?)
def _write_filedtrip_trials(spk, filepath):
    """
    Saves SpikeEpochs data to FieldTrip-compatible .mat file.

    Parameters
    ----------
    spk : SpikeEpochs
        SpikeEpochs object to save.
    filepath : str
        Path to the output .mat file.
    """
    from scipy.io import savemat

    # Initialize structure for FieldTrip format
    spikeTrials = {'trial': [], 'time': [], 'label': []}
    spikeTrials['dimord'] = '{chan}_lead_time_spike'
    has_metadata = spk.metadata is not None

    # Loop over cells
    n_units = spk.n_units()
    for cell_idx in range(n_units):
        spikeTrials['trial'].append(spk.trial[cell_idx] + 1)  # py -> mat idx
        spikeTrials['time'].append(spk.time[cell_idx])
        spikeTrials['label'].append(spk.cell_names[cell_idx])

    spikeTrials['trialtime'] = np.tile(spk.time_limits, (spk.n_trials, 1))

    spikeTrials = _waveform_to_ft(spk, spikeTrials)
    spikeTrials = _cellinfo_to_ft(spk, spikeTrials)

    if has_metadata:
        spikeTrials['trialinfo'] = spk.metadata.values
        spikeTrials['trialinfo_columns'] = spk.metadata.columns.tolist()

    # Save to a .mat file using scipy.io.savemat
    savemat(filepath, {'spike': spikeTrials})
    # use some logging later: print(f"Spike trials saved to {filepath}")


def _write_filedtrip_raw(spk, filepath):
    """
    Saves Spikes data to FieldTrip-compatible .mat file.

    Parameters
    ----------
    spk : Spikes
        Spikes object to save.
    filepath : str
        Path to the output .mat file.
    """
    from scipy.io import savemat

    # Initialize structure for FieldTrip format
    spikes = {'timestamp': [], 'label': []}
    spikes['dimord'] = '{chan}_lead_time_spike'
    spikes['hdr'] = dict(FileHeader=dict(Frequency=spk.sfreq))

    # Loop over cells
    n_units = spk.n_units()
    for cell_idx in range(n_units):
        spikes['timestamp'].append(spk.timestamps[cell_idx])
        spikes['label'].append(spk.cell_names[cell_idx])

    spikes = _waveform_to_ft(spk, spikes)
    spikes = _cellinfo_to_ft(spk, spikes)

    # Save to a .mat file using scipy.io.savemat
    savemat(filepath, {'spike': spikes})


def _waveform_to_ft(spk, spikeTrials):
    has_waveform = spk.waveform is not None
    has_waveform_time = spk.waveform_time is not None

    if has_waveform:
        n_units = spk.n_units()
        spikeTrials['waveform'] = np.empty(n_units, dtype='object')
        for cell_idx in range(n_units):
            # add "leads" dimention
            this_waveform = spk.waveform[cell_idx]
            if this_waveform is not None:
                this_waveform = this_waveform.T[None, :]
            else:
                this_waveform = np.array([])
            spikeTrials['waveform'][cell_idx] = this_waveform

    # save waveform_time if present
    if has_waveform_time:
        spikeTrials['waveform_time'] = spk.waveform_time

    return spikeTrials


def _cellinfo_to_ft(spk, spikeTrials):
    has_cellinfo = spk.cellinfo is not None
    if has_cellinfo:
        cellinfo = spk.cellinfo.to_records(index=False)
        spikeTrials['cellinfo'] = cellinfo

    return spikeTrials


# TODO: add progressbar?
def read_combinato(path, label=None, alignment='both'):
    '''Read spikes from combinato sorting output.

    Note that groups are read as single units, the class information is stored
    inside the Spikes object in `.cellinfo` attribute.

    Parameters
    ----------
    path : str
        Path to directory with combinato channel subdirectories with sorting
        results.
    label : str | None
        Read specific sorting labels (if multiple). Defaults to ``None``, which
        reads the first available label.
    alignment : str
        The alignment to read. Can be:
        * ``'pos'`` - read only positive alignment sorting results
        * ``'neg'`` - read only negative alignment sorting results
        * ``'both'`` - read both alignments
    '''
    import h5py

    if alignment == 'both':
        alignment = ['neg', 'pos']
    elif isinstance(alignment, str):
        alignment = [alignment]

    # currently we read only SU by default
    types_oi = [2]  # 2: SU; 1: MU; 0: artifact

    channel_dirs = os.listdir(path)
    group = list()
    align = list()
    channel = list()
    spike_data = {'timestamp': list(), 'waveform': list(), 'class': list(),
                  'distance': list()}

    for subdir in channel_dirs:
        has_content = False
        subdir_path = op.join(path, subdir)
        hdf5_file = op.join(subdir_path, 'data_' + subdir + '.h5')

        if op.isdir(subdir_path) and op.exists(hdf5_file):
            sort_dirs = os.listdir(subdir_path)

            for pol in alignment:
                subdir_prefix = f'sort_{pol}_'
                pol_subdirs = [f for f in sort_dirs if subdir_prefix in f]

                if len(pol_subdirs) == 0:
                    continue

                if label is None:
                    labels = np.unique([dr.split('_')[2]
                                       for dr in pol_subdirs])
                    if labels.shape[0] > 1:
                        msg = ('Multiple sorting labels present, please provi'
                               'de a sorting label to read results from. Found'
                               f' the following labels in {subdir} directory: '
                                ', '.join(labels))
                        raise RuntimeError(msg)
                    else:
                        label = str(labels[0])

                pol_subdir = f'sort_{pol}_{label}'
                full_sorting_path = op.join(subdir_path, pol_subdir,
                                            'sort_cat.h5')
                if not op.exists(full_sorting_path):
                    continue

                sorting_file = h5py.File(full_sorting_path, 'r')

                types = np.asarray(sorting_file['types'])
                # find SUs (or SUs and MUs)
                is_SU = np.isin(types[:, -1], types_oi)

                if not is_SU.any():
                    continue

                if not has_content:
                    spikes_file = h5py.File(hdf5_file, 'r')

                has_content = True
                groups_oi = types[is_SU, 0]
                groups = np.asarray(sorting_file['groups'])
                groups_sel = np.insin(groups[:, 1], groups_oi)
                groups = groups[groups_sel, :]

                spike_classes = np.asarray(sorting_file['classes'])
                spike_indices = np.asarray(sorting_file['index'])
                spike_distance = np.asarray(sorting_file['distance'])
                sorting_file.close()

                times = np.asarray(spikes_file[pol]['times'])
                waveforms = np.asarray(spikes_file[pol]['spikes'])

                for grp in groups_oi:
                    msk = groups[:, 1] == grp
                    this_classes = groups[msk, 0]
                    class_msk = np.isin(spike_classes, this_classes)

                    idx = spike_indices[class_msk]
                    # some groups can be empty with label that was not updated
                    if len(idx) > 0:
                        group.append(grp)
                        align.append(pol)
                        channel.append(subdir)

                        # Combinato times are in ms, but we turn this
                        # to 1 microsecond timestamps used by Neuralynx
                        spike_data['timestamp'].append(times[idx] * 1000)
                        spike_data['waveform'].append(waveforms[idx, :])

                        spike_data['class'].append(spike_classes[class_msk])
                        spike_data['distance'].append(
                            spike_distance[class_msk])

        if has_content:
            spikes_file.close()

    # organize into one Spikes object
    cellinfo = pd.DataFrame(data={'channel': channel, 'alignment': align,
                                  'group': group})
    spikes = Spikes(spike_data['timestamp'], sfreq=1e6, cellinfo=cellinfo,
                    waveform=spike_data['waveform'])
    return spikes


def read_osort(path, waveform=True, channels='all', format='mm',
               progress=True, use_usenegative=False):
    '''Read osort sorting results.

    The mm format can be obtained using updateSORTINGresults_mm matlab
    function from _psy_screening (located in
    ``psy_screenning-/helpers/sorting_utils``).

    Parameters
    ----------
    path : str
        Path to the directory with the ``.mat`` files obtained with
        ``updateSORTINGresults_mm``. It can also be one file with all the units
        in it.
    waveform : bool
        Whether to also read the waveforms. Waveforms typically take the most
        memory so it may be worth turning it off for fast reading. Defaults
        to ``True``.
    channels : str | list
        Channels to read. Defaults to ``'all'``, which reads all channels.
    format : str
        Osort file format. Can be:
        * ``'standard'`` - the default osort format
        * ``'mm'`` - format with cleaned up variable names and with alignment
            information
    progress : bool
        Whether to show progress bar. Defaults to ``True``.
    use_usenegative : bool
        Whether to use the ``usenegative`` field to select units. Defaults to
        ``False``. Works only with the ``'standard'`` format.

    Returns
    -------
    spk : pylabianca.spikes.Spikes
        Spikes object.
    '''
    from scipy.io import loadmat
    if progress:
        from tqdm import tqdm

    assert op.exists(path), 'Path/file does not exist.'
    good_format = format in ['standard', 'mm']
    if not good_format:
        msg = f'Unrecognized format "{format}".'
        raise ValueError(msg)

    if use_usenegative and format == 'mm':
        raise ValueError('The "usenegative" field is not available in the '
                         'mm format. mm format keeps the selected units only '
                         'during the export.')

    one_file = not op.isdir(path)

    if not one_file:
        files = [f for f in os.listdir(path) if f.endswith('.mat')]
        files.sort()

        # select files based on channels and format
        if not channels == 'all':
            if isinstance(channels, str):
                channels = [channels]
            channels_check = [ch + '_' for ch in channels]
            check_channel = lambda f: any([ch in f for ch in channels_check])
            files = [f for f in files if check_channel(f)]
        if format == 'mm':
            files = [f for f in files if 'mm_format' in f]
    else:
        full_path = Path(path)
        path = full_path.parent
        files = [full_path.name]

    cluster_id, alignment, threshold, channel = [list() for _ in range(4)]
    timestamp = list()
    waveforms = list() if waveform else None
    ignore_cluster = [0, 1, 99999999]

    if format == 'mm':
        # TEMP FIX: older exporting function had a spelling error
        correct_field = 'alignment' if one_file else ['aligment', 'alignment']
        var_names = None
        read_vars = ['cluster_id', 'threshold', 'channel', 'timestamp']
        if waveform:
            read_vars.append('waveform')

        translate = {var: var for var in read_vars}
        if one_file:
            var_names = read_vars + [correct_field]
    else:
        var_names = ['assignedNegative', 'newTimestampsNegative']
        # var_names = ['assignedNegative', '', '', 'newTimestampsNegative']
        if waveform:
            var_names.append('newSpikesNegative')
        if use_usenegative:
            var_names.append('useNegative')

        translate = {'cluster_id': 'assignedNegative',
                     'timestamp': 'newTimestampsNegative',
                     'waveform': 'newSpikesNegative'}

        # make sure path is a pathlib object
        path = Path(path) if not isinstance(path, Path) else path

        # check if threshold and alignment can be read from the path
        try:
            thresh_str = path.stem
            thresh = float(thresh_str)
        except:
            thresh = np.nan

        try:
            algn = path.parent.stem
            assert algn in ['min', 'max', 'mixed']
        except:
            algn = 'unknown'

    iter_over = tqdm(files) if progress else files
    for fname in iter_over:
        file_path = op.join(path, fname)
        data = loadmat(file_path, squeeze_me=False, variable_names=var_names)

        # make sure the correct format was specified - if not,
        # there will be no variables in the data object - throw an
        # informative error then
        clst_id_field = translate['cluster_id']
        if fname == files[0] and clst_id_field not in data:
            msg = (f'Could not find the "{clst_id_field}" field in the file '
                   f'{file_path} using the "{format}" format. Make sure the '
                   'correct format was specified.')
            raise ValueError(msg)

        this_cluster_id = data[clst_id_field].astype('int64')

        if format == 'mm' and isinstance(correct_field, list):
            # TEMP FIX: older exporting function had a spelling error
            correct_field = [field for field in correct_field
                             if field in data][0]
            var_names = read_vars + [correct_field]

        if format == 'mm':
            cluster_id.append(this_cluster_id)
            channel.append([x[0][0] for x in data['channel']])
            alignment.append([x[0][0] for x in data[correct_field]])
            threshold.append(data['threshold'].astype('float32'))

            # unpack to list (and then use .extend)
            timestamp.extend([x[0][0] for x in data['timestamp']])

            # trim first X waveform timesamples (interpolation artifact) ?
            if waveform:
                waveforms.extend([x[0] for x in data['waveform']])
        else:
            # find cluster ids
            if not use_usenegative:
                cluster_ids = np.unique(this_cluster_id)
                msk = np.isin(cluster_ids, ignore_cluster)
                if msk.any():
                    cluster_ids = cluster_ids[~msk]
            else:
                cluster_ids = data['useNegative'].ravel().astype('int64')
            cluster_id.append(cluster_ids)

            # find channel
            ch_name = fname.split('_')[0]
            ch_name = ch_name.replace('CSC', '')

            # split timestamps (and waveforms) into units
            this_timestamps = data[translate['timestamp']]
            for clst_id in cluster_ids:
                unit_msk = this_cluster_id == clst_id
                timestamp.append(this_timestamps[unit_msk])

                alignment.append(algn)
                threshold.append(thresh)
                channel.append(ch_name)

                if waveform:
                    # unit_msk is 1 x n_spikes so here we ignore the
                    # singleton dimension
                    waveforms.append(data[translate['waveform']][unit_msk[0]])

    cluster_id = np.concatenate(cluster_id)
    if format == 'mm':
        alignment = np.concatenate(alignment)
        threshold = np.concatenate(threshold)
        channel = np.concatenate(channel)

        cluster_id = cluster_id[:, 0]
        threshold = threshold[:, 0]

    cellinfo = pd.DataFrame(dict(channel=channel,
                                 cluster=cluster_id,
                                 alignment=alignment,
                                 threshold=threshold))

    # the sfreq here refers to timestamp frequency, not the sampling frequency
    # of the signal
    if waveform:
        n_samples = waveforms[0].shape[1]
        samples_per_ms = 100
        waveform_time = np.arange(n_samples) / samples_per_ms
    else:
        waveform_time = None

    return Spikes(timestamp, sfreq=1e6, cellinfo=cellinfo, waveform=waveforms,
                  waveform_time=waveform_time)


def read_events_neuralynx(path, events_file='Events.nev', format='dataframe',
                          first_timestamp_from='CSC130.ncs'):
    '''Read neuralynx events file as a simple array or dataframe.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the folder containing neuralynx files.
    events_file : str
        Name of the file containing events. ``'Events.nev'`` by default.
    format : "dataframe" | "mne"
        How to format the events:
        * ``"dataframe"`` - dataframe in BIDS events format
        * ``"mne"`` - array in mne-python format
        defaults to ``"dataframe"``.
    first_timestamp_from : str
        Name of the file to take first timestamp from. ``'CSC130.ncs'`` by
        default. Not used when ``False`` or ``format`` is ``"mne"``.

    Returns
    -------
    events : pandas.DataFrame | numpy.ndarray
        If ``format ="dataframe"``: dataframe containing event times (wrt the
        recording start), trigger values and timestamps. If ``format="mne"``:
        n_events by 3 array in mne-python format (first column - timestamps,
        last columns - trigger values).
    '''
    from .neuralynx_io import load_nev, load_ncs

    events_path = op.join(path, events_file)
    nev = load_nev(events_path)

    # take all trigger timestamps
    event_timestamps = nev['events']['TimeStamp'].astype('int64')

    # take only non-zero event triggers
    triggers = nev['events']['ttl']
    nonzero = triggers > 0
    event_timestamps = event_timestamps[nonzero]
    triggers = triggers[nonzero]

    n_events = event_timestamps.shape[0]
    if format == 'dataframe':
        # take first timestamp of the recording from one of the files
        if first_timestamp_from:
            ncs_path = op.join(path, first_timestamp_from)
            ncs = load_ncs(ncs_path)
            first_sample = ncs['time'][0].astype('int64')
            last_sample = ncs['time'][-1].astype('int64')
            del ncs

        # prepare dataframe
        columns = ['start', 'duration', 'type', 'trigger', 'timestamp']
        if not first_timestamp_from:
            columns.pop(2)

        indices = np.arange(0, n_events + (1 if first_timestamp_from else 0))
        events = pd.DataFrame(columns=columns, index=indices)

        # first row is a special case - info about first timestamp
        # of the recording
        if first_timestamp_from:
            events.loc[0, 'start'] = 0.
            events.loc[0, 'duration'] = 'n/a'
            events.loc[0, 'type'] = 'start'
            events.loc[0, 'trigger'] = -1
            events.loc[0, 'timestamp'] = first_sample
            start, end = 1, n_events
            starts = (event_timestamps - first_sample) / 1e6
        else:
            start, end = 0, n_events - 1
            starts = np.empty(n_events)
            starts.fill(np.nan)

        # the rest is just copying data to the dataframe
        events.loc[start:end, 'start'] = starts
        events.loc[start:end, 'duration'] = 'n/a'
        events.loc[start:end, 'trigger'] = triggers
        events.loc[start:end, 'timestamp'] = event_timestamps

        if first_timestamp_from:
            events.loc[start:end, 'type'] = 'trigger'

            # last timestamp
            events.loc[end + 1, 'start'] = (last_sample - first_sample) / 1e6
            events.loc[end + 1, 'duration'] = 'n/a'
            events.loc[end + 1, 'type'] = 'end'
            events.loc[end + 1, 'trigger'] = -1
            events.loc[end + 1, 'timestamp'] = last_sample

        events = events.infer_objects()

    elif format == 'mne':
        events = np.zeros((n_events, 3), dtype='int64')
        events[:, 0] = event_timestamps
        events[:, -1] = triggers
    else:
        raise ValueError(f'Unknown format "{format}".')

    return events


def read_events_neo(reader, format='mne'):
    '''Read events from a neo reader connected to a file.

    Parameters
    ----------
    reader : neo.io.BaseIO
        Neo reader connected to a file.
    format : "mne"
        How to format the events:
        * ``"mne"`` - array in mne-python format

        At the moment only ``"mne"`` format is supported.

    Returns
    -------
    events : numpy.ndarray
        n_events by 3 array in mne-python format (first column - timestamps,
        last columns - trigger values).
    '''
    assert format == 'mne', 'Only "mne" format is supported at the moment.'
    event_chan = reader.header['event_channels']
    n_event_ch = len(event_chan)

    events = [reader.get_event_timestamps(event_channel_index=idx)
              for idx in range(n_event_ch)]

    # for mne format, we only save events with event_id:
    event_times = list()
    event_ids = list()

    for timestamp, _, event_id in events:
        event_unique = np.unique(event_id)
        if len(event_unique) > 0 and not (event_unique == '').all():
            event_times.append(np.asarray(timestamp))
            event_id = np.asarray(event_id).astype(int)
            event_ids.append(event_id)

    n_elements = len(event_times)
    if n_elements == 0:
        # no events to return - raise error
        raise RuntimeError('No events in the file.')
    elif n_elements == 1:
        event_times = event_times[0]
        event_ids = event_ids[0]
    else:
        event_times = np.concatenate(event_times)
        event_ids = np.concatenate(event_ids)

        # sort by trigger time
        idx = np.argsort(event_times)
        event_times = event_times[idx]
        event_ids = event_ids[idx]

    # construct MNE-Python triggers array:
    n_events = len(event_times)
    events = np.zeros((n_events, 3), dtype=event_times.dtype)
    events[:, 0] = event_times
    events[:, -1] = event_ids

    return events


def read_events_plexon_nex(path, format='mne'):
    import neo
    reader = neo.io.NeuroExplorerIO(filename=path)
    return read_events_neo(reader, format=format)


def read_spikes_neo(reader, waveform=True, min_spikes=10):
    spike_info = reader.header['spike_channels']
    n_units = spike_info.shape[0]
    cell_info = pd.DataFrame(spike_info)

    timestamp_frate = cell_info.wf_sampling_rate.values.copy()
    tstmp_uni = np.unique(timestamp_frate[timestamp_frate > 0])
    assert len(tstmp_uni) == 1
    timestamp_frate = tstmp_uni[0]

    use_cells = np.zeros(n_units, dtype=bool)
    waveforms = list() if waveform else None
    timestamps = list()

    for ch_idx in range(n_units):
        tmstmp = reader.get_spike_timestamps(spike_channel_index=ch_idx)
        if min_spikes > 0:
            n_spk = len(tmstmp)
            if n_spk < min_spikes:
                continue

        timestamps.append(np.asarray(tmstmp))
        use_cells[ch_idx] = True

        if waveform:
            wvfm = reader.get_spike_raw_waveforms(spike_channel_index=ch_idx)
            has_wvfm = wvfm is not None

            if has_wvfm:
                wvfm_size = wvfm.shape
                if len(wvfm_size) == 3:
                    # middle dim is the lead dimension
                    n_leads = wvfm_size[1]
                    if n_leads > 1:
                        # warn that we take only the first one
                        warn('Currently if there are multiple leads only the '
                             'waveforms corresponding to the first one are '
                             'read.')
                    wvfm = wvfm.squeeze(axis=1)
                else:
                    wvfm = np.asarray(wvfm)

                # add to waveforms list:
                waveforms.append(wvfm * cell_info.loc[ch_idx, 'wf_gain'])
            else:
                waveforms.append(wvfm)

    cell_info = cell_info.loc[use_cells, :]
    cell_names = cell_info.name.values.copy()

    if waveform:
        waveform_samples = np.array([x.shape[-1] for x in waveforms
                                    if x is not None])
        assert (waveform_samples[0] == waveform_samples).all()
        waveform_samples = waveform_samples[0]
        waveform_time = np.arange(waveform_samples) / (timestamp_frate / 1_000)
    else:
        waveform_time = None

    spk = Spikes(
        timestamps, sfreq=timestamp_frate, cell_names=cell_names,
        cellinfo=cell_info, waveform=waveforms, waveform_time=waveform_time
    )
    return spk


# TODO: waveforms seem to be read incorrectly for fieldtrip
#       sample plexon nex - likely a bug in neo
def read_plexon_nex(path, waveform=True, min_spikes=10):
    import neo
    reader = neo.io.NeuroExplorerIO(filename=path)
    return read_spikes_neo(reader, waveform=waveform, min_spikes=min_spikes)


# CONSIDER: allow xarray output? format='xarray' and format='mne'?
def read_signal_neo(reader, channels='all'):
    import mne

    signal_header = reader.header['signal_channels']
    n_signal_channels = len(signal_header)
    dtypes = signal_header.dtype

    signal_array = [reader.get_analogsignal_chunk(stream_index=idx)[:, 0]
                    for idx in range(n_signal_channels)]
    signal_array = np.stack(signal_array, axis=0)

    if 'gain' in dtypes.fields:
        signal_array = signal_array * signal_header['gain'][:, None]

    if 'sampling_rate' in dtypes.fields:
        sampling_rate = signal_header['sampling_rate']
    else:
        sampling_rate = np.array(
            [reader.get_signal_sampling_rate(stream_index=idx)
            for idx in range(n_signal_channels)]
        )

    assert (sampling_rate[0] == sampling_rate).all()
    sfreq = sampling_rate[0]

    # make sure the signal is in Volts
    # translate 'units': 'mV' -> divide by 1_000; 'V' -> no change;
    #                    'uV' -> divide by 1e6
    if 'units' in dtypes.fields:
        translate_units = {'mV': 1 / 1_000, 'V': 1, 'uV': 1 / 1e6}
        units = signal_header['units']
        multips = np.array([translate_units[unit] for unit in units])
        if not (multips == np.ones(n_signal_channels)).all():
            signal_array *= multips[:, None]

    ch_names = signal_header['name'].tolist()

    # check offsets
    offsets = np.array([reader.get_signal_t_start(0, 0, stream_index=idx)
                        for idx in range(n_signal_channels)])
    assert (offsets[0] == offsets).all()
    first_sample = int(np.round(sfreq * offsets[0]))

    info = mne.create_info(ch_names, sfreq, ch_types='seeg')
    raw = mne.io.RawArray(signal_array, info, first_samp=first_sample)

    return raw


def read_signal_plexon_nex(path):
    import neo
    reader = neo.io.NeuroExplorerIO(filename=path)
    return read_signal_neo(reader)


# TODO - make sure we can save and write more cellinfo columns
#        (and unit names)
def _convert_spk_to_mm_matlab_format(spk):
    n_units = len(spk)

    data = dict()
    has_cellinfo = spk.cellinfo is not None

    if has_cellinfo:
        columns = spk.cellinfo.columns
        for fld in ['cluster_id', 'alignment', 'channel', 'threshold']:
            col_name = fld.split('_')[0]
            if col_name in columns:
                # extract values from dataframe column
                # and make sure it is N x 1
                data[fld] = spk.cellinfo.loc[:, col_name].values[:, None]

        # other cellinfo columns
        # TODO

    data['timestamp'] = np.empty((n_units, 1), dtype='object')
    data['waveform'] = np.empty((n_units, 1), dtype='object')

    for idx in range(n_units):
        data['timestamp'][idx, 0] = spk.timestamps[idx]
        data['waveform'][idx, 0] = spk.waveform[idx]

    return data


def _save_spk_to_mm_matlab_format(spk, path):
    from scipy.io import savemat

    data = _convert_spk_to_mm_matlab_format(spk)
    savemat(path, data)


# TODO: could use ``kind='auto'``, especially if we add support for
#       Unit or Session objects
def from_spiketools(inst, kind='trials'):
    '''Convert spiketools array representation to pylabianca objects.

    spiketools is another python package for spike analysis. You can find it
    here: https://spiketools.github.io/spiketools/

    Parameters
    ----------
    inst : list of arrays
        List of arrays containing spike times. Each array corresponds to one
        trial when ``kind='trials'``.
    kind : 'trials' | ...
        Kind of spiketools representation. Can be:
        * ``'trials'`` - list of arrays with spike times for each trial
        * ``'times'`` - single array with spike times
    '''
    if kind == 'times':
        assert inst.ndim == 1
        assert isinstance(inst, np.ndarray)
        assert all([isinstance(x, np.ndarray) for x in inst])
        inst = [inst]

    if kind == 'trials':
        assert isinstance(inst, list)
        assert all([isinstance(x, np.ndarray) for x in inst])
        assert all([x.ndim == 1 for x in inst])

    trial_ids = [np.ones(len(x), dtype=int) * idx
                    for idx, x in enumerate(inst)]
    trial_ids = np.concatenate(trial_ids)
    spike_times = np.concatenate(inst)

    spk_epochs = SpikeEpochs([spike_times], [trial_ids])
    return spk_epochs


def to_spiketools(spk_epochs, picks=None):
    '''Convert pylabianca SpikeEpochs to list of arrays.

    Parameters
    ----------
    spk_epochs : SpikeEpochs
        SpikeEpochs object to convert.
    picks : None | list of ints
        Which units to convert. If ``None``, all units are converted.

    Returns
    -------
    inst : list of arrays
        List of arrays containing spike times. Each array corresponds to one
        trial. When multiple picks are provided, the output is a list of lists
        of arrays (where outermost list elements correspond to units).
    '''
    msg = ('Only exporting SpikeEpochs to spiketools is supported. You can '
           'convert Spikes to SpikeEpochs beforehand by using '
           '``.to_epochs()`` method of Spikes.')
    assert isinstance(spk_epochs, SpikeEpochs), msg

    picks = _deal_with_picks(spk_epochs, picks)
    one_unit = len(picks) == 1

    unit_list = _to_arrays(spk_epochs, picks)

    if one_unit:
        return unit_list[0]
    else:
        return unit_list


def _to_arrays(spk_epochs, picks):

    max_trials = spk_epochs.n_trials
    unit_list = list()
    for idx in picks:
        trial_list = list()
        tri_limits, tri_ids = _get_trial_boundaries(spk_epochs, idx)
        tri_enum = 0
        for tri in range(max_trials):
            if tri in tri_ids:
                lim1, lim2 = tri_limits[tri_enum:tri_enum + 2]
                trial_list.append(spk_epochs.time[idx][lim1:lim2])
                tri_enum += 1
            else:
                trial_list.append(np.array([]))
        unit_list.append(trial_list)
    return unit_list


def _get_chan_num(chan):
    from numbers import Integral
    if isinstance(chan, Integral):
        return chan
    else:
        return int(''.join([char for char in chan if char.isdigit()]))


def add_region_from_channel_ranges(spk, channel_info, source_column='area',
                                   target_column='region',
                                   channel_column='channel'):
    '''Add brain region information to Spikes from channel ranges info table.

    Parameters
    ----------
    spk : Spikes | SpikeEpochs
        Spikes object whose cell metadata (``.cellinfo``) will be filled
        with information from the table. The ``.cellinfo`` is expected to
        contain channel information in column ``'channel'``. The channel
        information should be either an integer or string, but for strings
        non-numeric values are removed and the remaining numeric values are
        concatenated and turned into integer.
    channel_info : pandas.DataFrame
        Dataframe containing brain region info for specified channel ranges.
        The expected format of this table is the following:
        * ``'channel start'`` column specifying starting values of a channel
          range (for example microwires belonging to the same Behnke-Fried
          electrode).
        * ``'channel end'`` column specifying ending values of a channel range.

        The column specifyng region for given channel range chan be set using
        ``source_column`` argument.
    '''
    assert isinstance(spk.cellinfo, pd.DataFrame)
    assert 'channel' in spk.cellinfo.columns
    chans = spk.cellinfo.channel.unique()

    numeric_rows = [isinstance(x, (int, float))
                    for x in channel_info['channel start']]
    channel_info = channel_info.loc[numeric_rows, :]

    for chan in chans:
        chan_num = _get_chan_num(chan)
        msk = (channel_info['channel start'] <= chan_num) & (
            channel_info['channel end'] >= chan_num)
        region = channel_info.loc[msk, source_column].values[0]

        cell_msk = spk.cellinfo.channel == chan
        spk.cellinfo.loc[cell_msk, target_column] = region


def read_drop_info(path):
    '''Reads (channels, cluster id) pairs to drop from a text file.

    The text file should follow a structure:
    channel_name1: [cluster_id1, cluster_id2, ...]
    channel_name2: [cluster_id1, cluster_id2, ...]

    Parameters
    ----------
    path : str
        Path to the text file.

    Returns
    -------
    to_drop : list
        List of (channel, cluster_id) tuples representing all such pairs
        read from the text file.
    '''
    # read merge info
    with open(path) as file:
        text = file.readlines()

    # drop info is organized into channels / cluster ids
    to_drop = list()
    for line in text:
        channel = line.split(', ')[0]
        idx1, idx2 = line.find('['), line.find(']') + 1
        clusters = eval(line[idx1:idx2])
        for cluster in clusters:
            to_drop.append((channel, cluster))

    return to_drop
