import os
import os.path as op
from pathlib import Path

import numpy as np
import pandas as pd

from .spikes import SpikeEpochs


def prepare_gammbur_metadata(df):
    '''Prepare behavioral data from GammBur.
    Name columns apropriately and set their dtypes.
    '''
    df.columns = ['dig1', 'dig2', 'dig3', 'ifcorrect', 'load', 'ifout',
                  'probe', 'RT']

    int_cols = ['dig1', 'dig2', 'dig3', 'load', 'probe']
    col_types = {col: 'int' for col in int_cols}
    col_types.update({col: 'bool' for col in ['ifcorrect', 'ifout']})
    df = df.astype(col_types)

    df['RT'] = df['RT'] / 1000
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
        import sarna

        dropbox_dir = Path(sarna.proj.find_dropbox())
        proj_dir = dropbox_dir / 'PROJ' / 'Labianka' / 'GammBur'
        data_dir = proj_dir / 'cleandata'
        assert op.isdir(data_dir)
        fls = os.listdir(data_dir)

        subj_id_txt = '{:02d}'.format(subject_id)
        fname = [f for f in fls if f.startswith(subj_id_txt)][0]
        fname = data_dir / fname

    if kind == 'spikes':
        return _read_spikes_gammbur(fname)
    elif kind == 'lfp':
        return _read_lfp_gammbur(fname)
    else:
        raise ValueError('The data kind to read has to be "spikes" or "lfp"')


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
        necessay to specify. ``'spike'`` by default.

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

    epochs = mne.io.read_epochs_fieldtrip(fname, info, data_name='lfp')
    epochs.metadata = prepare_gammbur_metadata(epochs.metadata)
    return epochs
