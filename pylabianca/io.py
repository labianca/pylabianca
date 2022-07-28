import os
import os.path as op
from pathlib import Path

import numpy as np
import pandas as pd

from .spikes import SpikeEpochs, Spikes


# TODO - consider moving gammbur specific code to GammBur...
def prepare_gammbur_metadata(df, trial_indices=None):
    '''Prepare behavioral data from GammBur.

    This function is specific to GammBur project. It names columns
    appropriately and sets their dtypes.
    '''
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    # set column names
    df.columns = ['dig1', 'dig2', 'dig3', 'ifcorrect', 'load', 'ifout',
                  'probe', 'RT']

    # set dtypes
    int_cols = ['dig1', 'dig2', 'dig3', 'load', 'probe']
    col_types = {col: 'int' for col in int_cols}
    col_types.update({col: 'bool' for col in ['ifcorrect', 'ifout']})
    df = df.astype(col_types)

    # set RT to seconds
    df['RT'] = df['RT'] / 1000

    if trial_indices is None:
        n_trials = df.shape[0]
        trial_indices = np.arange(n_trials)

    df.loc[:, 'trial'] = trial_indices
    return df


def read_gammbur(subject_id=None, fname=None, kind='spikes', verbose=True):
    '''Read GammBur fieldtrip data format straight from the .mat file.

    Parameters
    ----------
    fname : str | pathlib.Path
        File name or full filepath to the ``.mat`` file.
    kind : str
        The data kind to read. Currently ``'spikes'`` and ``'lfp'`` are
        supported.
    verbose : bool
        Verbosity level.

    Returns
    -------
    data : mne.Epochs | SpikeEpochs
        Object containing the data.
    '''
    if fname is None and subject_id is not None:
        fname = find_file_name_gammbur(subject_id)
    if kind == 'spikes':
        return _read_spikes_gammbur(fname)
    elif kind == 'lfp':
        return _read_lfp_gammbur(fname, verbose=verbose)
    else:
        raise ValueError('The data kind to read has to be "spikes" or "lfp"')


def find_file_name_gammbur(subject_id, data_dir='cleandata'):
    """Find the GammBur file name for a given subject."""
    import sarna

    dropbox_dir = Path(sarna.proj.find_dropbox())
    proj_dir = dropbox_dir / 'PROJ' / 'Labianka' / 'GammBur'
    data_dir = proj_dir / data_dir
    assert op.isdir(data_dir)
    fls = os.listdir(data_dir)

    subj_id_txt = '{:02d}'.format(subject_id)
    fname = [f for f in fls if f.startswith(subj_id_txt)][0]
    fname = data_dir / fname
    return fname


def read_raw_gammbur(subject_id=None, fname=None):
    """Read raw GammBur spikes data."""
    if fname is None and subject_id is not None:
        fname = find_file_name_gammbur(subject_id, data_dir='cleandataraw')
    spk, events = read_raw_spikes(fname, data_name='ft_format')
    spk.metadata = prepare_gammbur_metadata(spk.metadata)
    return spk, events


# TODO
# - [ ] read waveform too...
def read_spikes(fname, data_name='spike'):
    '''Read fieldtrip SpikeTrials format.

    Parameters
    ----------
    fname : str | pathlib.Path
        Path to the file to read.
    data_name : str
        The name of the saved variable - this can be arbitrary so it is
        necessary to specify. ``'spike'`` by default.

    Returns
    -------
    spk : SpikeEpochs
        SpikeEpochs object.
    '''
    from scipy.io import loadmat

    data = loadmat(fname, squeeze_me=True, variable_names=data_name)[data_name]
    cell_names = data['label'].item()
    time = data['time'].item()
    trial = data['trial'].item() - 1
    trialtime = data['trialtime'].item()
    trialinfo = data['trialinfo'].item()

    msg = 'All trials have to be of the same length'
    assert (trialtime == trialtime[[0]]).all(), msg

    n_trials = trialtime.shape[0]
    time_limits = trialtime[0]
    trialinfo = pd.DataFrame(trialinfo)

    # create SpikeEpochs
    spk = SpikeEpochs(time, trial, time_limits, n_trials=n_trials,
                      metadata=trialinfo, cell_names=cell_names)
    spk.filename = fname
    return spk


def _read_spikes_gammbur(fname):
    '''GammBur-specific function that reads spikes and formats metadata.'''
    spikes = read_spikes(fname, data_name='spikes')
    spikes.metadata = prepare_gammbur_metadata(spikes.metadata)
    return spikes


def _read_lfp_gammbur(fname, verbose=True):
    '''GammBur-specific function that reads lfp data and formats metadata.'''
    import mne
    from scipy.io import loadmat

    sfreq = 500  # assumed LFP sampling frequency
    ch_names = ['dlpfc0{}'.format(idx) for idx in range(1, 5)]
    ch_names += ['hippo01', 'hippo02']
    info = mne.create_info(ch_names, sfreq, ch_types='seeg', verbose=verbose)

    matfile = loadmat(fname, squeeze_me=True, simplify_cells=True)
    has_lfp = ('lfp' in matfile) and (len(matfile['lfp']) > 0)

    if has_lfp:
        epochs = mne.io.read_epochs_fieldtrip(fname, info, data_name='lfp')
        tri_idx = _prepare_trial_indices(
            epochs, matfile['removed_tri_lfp'] - 1)
        epochs.metadata = prepare_gammbur_metadata(
            epochs.metadata, trial_indices=tri_idx)
        return epochs
    else:
        # given file does not contain lfp
        return None


def read_raw_spikes(fname, data_name='spikes'):
    '''Read raw spikes fieldtrip format.

    Parameters
    ----------
    fname : str
        Filename / path to the file.
    data_name : str
        Name of the variable stored in the .mat file.

    Returns
    -------
    spk : Spikes
        Spikes object.
    events : np.ndarray | None
        If ``.events`` field is present in the mat file ``events`` contain
        64 bit numpy array of the shape n_events x 2. Otherwise it is ``None``.
    '''
    from scipy.io import loadmat
    data = loadmat(fname, squeeze_me=True, variable_names=data_name)[data_name]

    cell_names = data['label'].item()
    timestamps = data['timestamp'].item()
    trialinfo = data['trialinfo'].item()
    fields = data['cellinfo'].item().dtype.names

    if 'cellinfo' in data.dtype.names:
        data_dct = dict()
        for fld in fields:
            data_dct[fld] = data['cellinfo'].item()[fld].item()
        cellinfo = pd.DataFrame(data_dct)
    else:
        cellinfo = None

    sfreq = data['hdr'].item()['FileHeader'].item()['Frequency'].item()
    trialinfo = data['trialinfo'].item()

    if 'events' in data.dtype.names:
        events = data['events'].item().astype('int64')
    else:
        events = None

    # create Spikes
    spk = Spikes(timestamps, sfreq, cell_names=cell_names,
                 metadata=trialinfo, cellinfo=cellinfo)
    return spk, events


def _prepare_trial_indices(epochs, removed_idx):
    n_removed = len(removed_idx)
    n_all_tri = epochs.metadata.shape[0] + n_removed
    tri_idx = np.arange(n_all_tri)
    tri_idx = np.delete(tri_idx, removed_idx)
    return tri_idx


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


# TODO: add option to resample (and trim?) the waveforms
# TODO: add option to read the standard osort format (``format='standard'``)
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
    ignore_cluster = [0, 99999999]

    if format == 'mm':
        # TEMP FIX: older exporting function had a spelling error
        correct_field = 'alignment' if one_file else ['aligment', 'alignment']
        var_names = None
        read_vars = ['cluster_id', 'threshold', 'channel', 'timestamp']
        if waveform:
            read_vars.append('waveform')

        translate = {var: var for var in read_vars}
        if not one_file:
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

    iter_over = tqdm(files) if progress else files
    for fname in iter_over:
        file_path = op.join(path, fname)
        data = loadmat(file_path, squeeze_me=False, variable_names=var_names)

        if format == 'mm' and isinstance(correct_field, list):
            # TEMP FIX: older exporting function had a spelling error
            correct_field = [field for field in correct_field
                             if field in data][0]
            var_names = read_vars + [correct_field]

        this_cluster_id = data[translate['cluster_id']].astype('int64')
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
            if ignore_cluster in cluster_ids:
                msk = np.in1d(cluster_ids, ignore_cluster)
                cluster_ids = cluster_ids[~msk]
            cluster_id.append(cluster_ids)

            # find channel
            ch_name = fname.split('_')[0]
            ch_name = ch_name.replace('CSC', '')

            thresh = float(path.stem)
            algn = path.parent.stem

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

    return Spikes(timestamp, sfreq=1e6, cellinfo=cellinfo, waveform=waveforms)


def read_neuralynx_events(path, events_file='Events.nev', format='dataframe',
                          first_timestamp_from='CSC130.ncs'):
    '''Turn neuralynx events file to a simple dataframe.

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
    import neuralynx_io as ni

    events_path = op.join(path, events_file)
    nev = ni.neuralynx_io.load_nev(events_path)

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
            ncs = ni.neuralynx_io.load_ncs(ncs_path)
            first_sample = ncs['time'][0].astype('int64')
            del ncs

        # prepare dataframe
        columns = ['start', 'duration', 'type', 'trigger', 'timestamp']
        if first_timestamp_from:
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
            start,end = 0, n_events - 1

        # the rest is just copying data to the dataframe
        starts = (event_timestamps - first_sample) / 1e6
        events.loc[start:end, 'start'] = starts
        events.loc[start:end, 'duration'] = 'n/a'
        events.loc[start:end, 'trigger'] = triggers
        events.loc[start:end, 'timestamp'] = event_timestamps

        if first_timestamp_from:
            events.loc[start:end, 'type'] = 'trigger'

        events= events.infer_objects()
    elif format == 'mne':
        events = np.zeros((n_events, 3), dtype='int64')
        events[:, 0] = event_timestamps
        events[:, -1] = triggers
    else:
        raise ValueError(f'Unknown format "{format}".')

    return events


def add_region_from_channels_table(spk, channel_info):
    '''Add brain region information to Spikes from channel info excel table.

    Parameters
    ----------
    spk : Spikes | SpikeEpochs
        Spikes object whose cell metadata (``.cellinfo``) should be filled
        with brain region information from the table.
    channel_info : pandas.DataFrame
        Dataframe containing brain region info for specified channel ranges.
    '''
    chans = spk.cellinfo.channel.unique()

    for chan in chans:
        chan_num = int(''.join([char for char in chan if char.isdigit()]))
        msk = (channel_info['channel start'] <= chan_num) & (
            channel_info['channel end'] >= chan_num)
        region = channel_info.loc[msk, 'area'].values[0]

        cell_msk = spk.cellinfo.channel == chan
        spk.cellinfo.loc[cell_msk, 'region'] = region
