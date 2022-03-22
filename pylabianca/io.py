import os
import os.path as op
from pathlib import Path

import numpy as np
import pandas as pd

from .spikes import SpikeEpochs, Spikes


def prepare_gammbur_metadata(df, trial_indices=None):
    '''Prepare behavioral data from GammBur.
    Name columns appropriately and set their dtypes.
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
        tri_idx = _prepare_trial_inices(epochs, matfile['removed_tri_lfp'] - 1)
        epochs.metadata = prepare_gammbur_metadata(epochs.metadata,
                                                   trial_indices=tri_idx)
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


def _prepare_trial_inices(epochs, removed_idx):
    n_removed = len(removed_idx)
    n_all_tri = epochs.metadata.shape[0] + n_removed
    tri_idx = np.arange(n_all_tri)
    tri_idx = np.delete(tri_idx, removed_idx)
    return tri_idx


# TODO: add progressbar?
# TODO: waveforms!
def read_combinato(path, label=None, alignment='both'):
    '''Read spikes from combinato sorting output.

    Note that groups are read as single units, the class information is stored
    inside the Spikes object.

    Parameters
    ----------
    '''
    import h5py

    if alignment == 'both':
        alignment = ['neg', 'pos']

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
