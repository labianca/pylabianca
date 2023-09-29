import os
import os.path as op
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd

from .spikes import SpikeEpochs, Spikes


# TODO - trialinfo columns ...
def read_filedtrip(fname, data_name='spike', kind='raw', waveform=True):
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

    if 'trialinfo' in data.dtype.names:
        trialinfo = data['trialinfo'].item()
        trialinfo = pd.DataFrame(trialinfo)
    else:
        trialinfo = None

    if 'cellinfo' in data.dtype.names:
        data_dct = dict()
        fields = data['cellinfo'].item().dtype.names
        for fld in fields:
            data_dct[fld] = data['cellinfo'].item()[fld].item()
        cellinfo = pd.DataFrame(data_dct)
    else:
        cellinfo = None

    if kind in ['trials', 'epochs']:
        spk = _read_ft_spikes_tri(data, cell_names, trialinfo, cellinfo,
                                  waveform, waveform_time)
    elif kind == 'raw':
        spk = _read_ft_spikes_raw(data, cell_names, cellinfo,
                                  waveform, waveform_time)

    spk.filename = fname
    return spk


# TODO
# - [ ] read waveform too...
def _read_ft_spikes_tri(data, cell_names, trialinfo, cellinfo, waveform,
                        waveform_time):
    '''Read fieldtrip SpikeTrials format.

    Returns
    -------
    spk : SpikeEpochs
        SpikeEpochs object.
    '''

    time = data['time'].item()
    trial = data['trial'].item() - 1
    trialtime = data['trialtime'].item()

    msg = 'All trials have to be of the same length'
    assert (trialtime == trialtime[[0]]).all(), msg

    n_trials = trialtime.shape[0]
    time_limits = trialtime[0]

    # create SpikeEpochs
    spk = SpikeEpochs(time, trial, time_limits, n_trials=n_trials,
                      metadata=trialinfo, cell_names=cell_names,
                      cellinfo=cellinfo, waveform=waveform,
                      waveform_time=waveform_time)

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
    sfreq = data['hdr'].item()['FileHeader'].item()['Frequency'].item()

    if 'waveform' in data.dtype.names:
        waveforms = data['waveform'].item()
        new_waveforms = list()
        n_units = len(waveforms)
        for ix in range(n_units):
            this_waveform = waveforms[ix]

            if (isinstance(this_waveform, np.ndarray)
                and len(this_waveform) > 0):

                # we currently take only the first lead
                if this_waveform.ndim > 2:
                    this_waveform = this_waveform[0]
                elif this_waveform.ndim == 1:
                    # only one waveform, squeezed
                    this_waveform = this_waveform[:, None]
                new_waveforms.append(this_waveform.T)
            else:
                new_waveforms.append(None)

        waveform_length = [x.shape[1] for x in new_waveforms if x is not None]
        same_lengths = all(waveform_length[0] == x
                           for x in waveform_length[1:])

        if same_lengths:
            # waveform time in ms
            waveform_time = np.arange(waveform_length[0]) / (sfreq / 1000)
        else:
            warn('Not all waveforms have the same number of samples, '
                 'waveforms are therefore ignored. Got the following waveform'
                 f'lengths: {waveform_length}.')
            waveforms, waveform_time = None, None
    else:
        waveforms, waveform_time = None, None

    return new_waveforms, waveform_time


# TODO: add progressbar?
def read_combinato(path, label=None, alignment='both'):
    '''Read spikes from combinato sorting output.

    Note that groups are read as single units, the class information is stored
    inside the Spikes object.

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
                is_SU = np.in1d(types[:, -1], types_oi)

                if not is_SU.any():
                    continue

                if not has_content:
                    spikes_file = h5py.File(hdf5_file, 'r')

                has_content = True
                groups_oi = types[is_SU, 0]
                groups = np.asarray(sorting_file['groups'])
                groups_sel = np.in1d(groups[:, 1], groups_oi)
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
                    class_msk = np.in1d(spike_classes, this_classes)

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
               progress=True):
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

    Returns
    -------
    spk : pylabianca.spikes.Spikes
        Spikes object.
    '''
    from scipy.io import loadmat
    if progress:
        from tqdm import tqdm

    assert op.exists(path), 'Path/file does not exist.'
    assert format in ['standard', 'mm'], 'Unknown format.'
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
        this_cluster_id = data[translate['cluster_id']].astype('int64')

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
            cluster_ids = np.unique(this_cluster_id)
            msk = np.in1d(cluster_ids, ignore_cluster)
            if msk.any():
                cluster_ids = cluster_ids[~msk]
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
    if waveform is not None:
        n_samples = waveforms[0].shape[1]
        samples_per_ms = 100
        waveform_time = np.arange(n_samples) / samples_per_ms

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
        else:
            start, end = 0, n_events - 1

        # the rest is just copying data to the dataframe
        starts = (event_timestamps - first_sample) / 1e6
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

        events= events.infer_objects()

    elif format == 'mne':
        events = np.zeros((n_events, 3), dtype='int64')
        events[:, 0] = event_timestamps
        events[:, -1] = triggers
    else:
        raise ValueError(f'Unknown format "{format}".')

    return events


# TODO - make sure we can save and write more cellinfo columns
#        (and unit names)
def _convert_spk_to_mm_matlab_format(spk):
    n_units = len(spk)

    data = dict()
    for fld in ['cluster_id', 'alignment', 'channel', 'threshold']:
        col_name = fld.split('_')[0]

        # extract values from dataframe column and make sure it is N x 1
        data[fld] = spk.cellinfo.loc[:, col_name].values[:, None]

    # other cellinfo columns

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


def add_region_from_channels_table(spk, channel_info, source_column='area',
                                   target_column='region'):
    '''Add brain region information to Spikes from channel info excel table.

    Parameters
    ----------
    spk : Spikes | SpikeEpochs
        Spikes object whose cell metadata (``.cellinfo``) should be filled
        with brain region information from the table.
    channel_info : pandas.DataFrame
        Dataframe containing brain region info for specified channel ranges.
    '''
    assert isinstance(spk.cellinfo, pd.DataFrame)
    assert 'channel' in spk.cellinfo.columns
    chans = spk.cellinfo.channel.unique()

    numeric_rows = [isinstance(x, (int, float))
                    for x in channel_info['channel start']]
    channel_info = channel_info.loc[numeric_rows, :]

    for chan in chans:
        chan_num = int(''.join([char for char in chan if char.isdigit()]))
        msk = (channel_info['channel start'] <= chan_num) & (
            channel_info['channel end'] >= chan_num)
        region = channel_info.loc[msk, source_column].values[0]

        cell_msk = spk.cellinfo.channel == chan
        spk.cellinfo.loc[cell_msk, target_column] = region
