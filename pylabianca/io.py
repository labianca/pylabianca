import os
import os.path as op
from pathlib import Path

import numpy as np
import pandas as pd

from .spikes import SpikeEpochs, Spikes


def prepare_gammbur_metadata(df):
    '''Prepare behavioral data from GammBur.
    Name columns appropriately and set their dtypes.
    '''
    if isinstance(df, np.ndarray):
        df = pd.DataFrame(df)

    df.columns = ['dig1', 'dig2', 'dig3', 'ifcorrect', 'load', 'ifout',
                  'probe', 'RT']

    int_cols = ['dig1', 'dig2', 'dig3', 'load', 'probe']
    col_types = {col: 'int' for col in int_cols}
    col_types.update({col: 'bool' for col in ['ifcorrect', 'ifout']})
    df = df.astype(col_types)

    df['RT'] = df['RT'] / 1000

    n_trials = df.shape[0]
    df.loc[:, 'trial'] = np.arange(n_trials)
    return df


def read_gammbur(subject_id=None, fname=None, kind='spikes'):
    '''Read GammBur fieldtrip data format straight from the .mat file.

    Parameters
    ----------
    fname : str | pathlib.Path
        File name or full filepath to the ``.mat`` file.
    kind : str
        The data kind to read. Currently ``'spikes'`` and ``'lfp'`` are
        supported.

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
        return _read_lfp_gammbur(fname)
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


def _read_lfp_gammbur(fname):
    '''GammBur-specific function that reads lfp data and formats metadata.'''
    import mne

    sfreq = 500  # assumed LFP sampling frequency
    ch_names = ['dlpfc0{}'.format(idx) for idx in range(1, 5)]
    ch_names += ['hippo01', 'hippo02']
    info = mne.create_info(ch_names, sfreq, ch_types='seeg')

    try:
        epochs = mne.io.read_epochs_fieldtrip(fname, info, data_name='lfp')
        epochs.metadata = prepare_gammbur_metadata(epochs.metadata)
        return epochs
    except IndexError:
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
