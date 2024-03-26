import os
import os.path as op
from warnings import warn

import numpy as np
import pandas as pd


def _deal_with_picks(spk, picks):
    '''Deal with various formats in which ``picks`` can be passed.

    Parameters
    ----------
    spk : pylabianca.Spikes | pylabianca.SpikeEpochs
        Spikes or SpikeEpochs object.
    picks : int | str | list-like of int | list-like of str | None
        The units to pick.

    Returns
    -------
    picks : list
        List of indices of the picked cells.
    '''
    has_str = False
    if picks is None:
        # pick all cells by default
        picks = np.arange(len(spk.cell_names))
        return picks
    if isinstance(picks, (list, np.ndarray, pd.Series)):
        if len(picks) == 0:
            raise ValueError('No cells selected.')
        if isinstance(picks[0], str):
            # list / array of names
            is_str = [isinstance(x, str) for x in picks[1:]]
            has_str = all(is_str) or len(picks) == 1
        elif all([isinstance(picks[ix], (bool, np.bool_))
                  for ix in range(len(picks))]):
            # list / array of booleans
            picks = np.where(picks)[0]
        elif isinstance(picks, pd.Series):
            picks = picks.values
    if not isinstance(picks, (list, np.ndarray, pd.Series)):
        if isinstance(picks, str):
            has_str = True
        picks = [picks]
    if has_str:
        if isinstance(spk.cell_names, list):
            picks = [spk.cell_names.index(name) for name in picks]
        else:
            picks = [np.where(spk.cell_names == name)[0][0] for name in picks]
    return picks


# CONSIDER: changing the array dim order to: trials, cells, times
#           (mne-python-like)
# CHANGE name to something more general - it is now used for xarray and decoding
#        results (and more in the future)
# spike_epochs is used only when n_trials > 0 to inherit metadata
#              and to copy cellinfo
def _turn_spike_rate_to_xarray(times, frate, spike_epochs, cell_names=None,
                               tri=None, copy_cellinfo=True,
                               x_dim_name='time'):
    '''Turn spike rate data to xarray.

    Parameters
    ----------
    times : numpy array | str
        Vector of time points for which spike rate was calculated (middle
        time points for the time window used). Can also be a string
        describing the time window if static window was used.
    frate : numpy array
        Numpy array with firing rate, with the following dimensions:

        * 3d ``n_cells x n_trials x n_times`` (``cell_names`` has to be not
          None)
        * 2d ``n_cells x n_trials`` (``cell_names`` not None and ``times``
          as string)
        * 2d ``n_trials x n_times`` (``cell_names`` is None and ``times``
          is an array)
        * 2d ``n_cells x n_times`` (``cell_names`` is not None and ``times``
          is an array)

    spike_epochs : SpikeEpochs object
        SpikeEpochs object.
    cell_names : list-like of str | None
        Names of the picked cells. If not ``None`` then indicates that the
        first dimension of ``frate`` contains cells.
    tri : np.ndarray | None
        Array of trial indices. Use when the repetitions dimension of ``frate``
        array is not equivalent to trials, but at least some repetitions come
        from the same trial (for example - spikes within trials when using
        spike-centered windows). Passing ``tri`` allows to copy the trial
        metadata correctly.
    copy_cellinfo : bool
        Whether to copy ``spike_epochs.cellinfo`` to xarray.
    x_dim_name : str
        Name of the last dimension. Defaults to ``'time'``.

    Returns
    -------
    firing : xarray
        Firing rate xarray.
    '''
    import xarray as xr

    # later: consider having firing rate from many neurons...
    times_array = isinstance(times, np.ndarray)
    if frate.ndim == 3:
        n_trials = frate.shape[0] if cell_names is None else frate.shape[1]
    elif frate.ndim == 2:
        if cell_names is None:
            n_trials = frate.shape[0]
        else:
            if times_array:
                n_trials = 0
            else:
                n_trials = frate.shape[1]

    if n_trials > 0:
        dimname = 'trial' if tri is None else 'spike'
        coords = {dimname: np.arange(n_trials)}
        dims = [dimname]
    else:
        coords = dict()
        dims = list()

    attrs = None
    if isinstance(times, np.ndarray):
        dims.append(x_dim_name)
        coords[x_dim_name] = times
    else:
        attrs = {'timewindow': times}

    if cell_names is not None:
        assert frate.shape[0] == len(cell_names)
        dims = ['cell'] + dims
        coords['cell'] = cell_names

    if tri is not None:
        coords['trial'] = (dimname, tri)

    if n_trials > 0:
        coords = _inherit_metadata(
            coords, spike_epochs.metadata, dimname, tri=tri)

    if copy_cellinfo:
        if cell_names is not None and spike_epochs.cellinfo is not None:
            ch_idx = _deal_with_picks(spike_epochs, cell_names)
            for col in spike_epochs.cellinfo.columns:
                coords[col] = (
                    'cell', spike_epochs.cellinfo[col].iloc[ch_idx])

    firing = xr.DataArray(frate, dims=dims, coords=coords,
                          attrs=attrs)
    return firing


def _inherit_metadata(coords, metadata, dimname, tri=None):
    if metadata is not None:
        for col in metadata.columns:
            if tri is None:
                coords[col] = (dimname, metadata[col])
            else:
                coords[col] = (dimname, metadata[col].iloc[tri])
    return coords


def _symmetric_window_samples(winlen, sfreq):
    '''Returns a symmetric window of given length.'''
    half_len_smp = int(np.round(winlen / 2 * sfreq))
    win_smp = np.arange(-half_len_smp, half_len_smp + 1)
    return win_smp, half_len_smp


def _gauss_kernel_samples(window, gauss_sd):
    '''Returns a gaussian kernel given window and sd in samples.'''
    from scipy.stats.distributions import norm
    kernel = norm(loc=0, scale=gauss_sd)
    kernel = kernel.pdf(window)
    return kernel


# ENH: allow for asymmetric windows (like in fieldtrip)
# ENH: inherit metadata from spike_epochs?
def spike_centered_windows(spk, arr, pick=None, time=None, sfreq=None,
                           winlen=0.1):
    '''Cut out windows from signal centered on spike times.

    Parameters
    ----------
    spk : pylabianca.SpikeEpochs
        Spike epochs object.
    arr : np.ndarray | xarray.DataArray | mne.Epochs
        Array with the signal to be cut out. The dimensions should be
        ``n_trials x n_channels x n_times``.
    pick : int | str
        Cell providing spikes centering windows.
    time : None | str | np.ndarray
        Time information for ``arr`` input:
        * if ``arr`` is an ``np.ndarray`` then ``time`` should be an array
          of time points for each sample of the time dimension.
        * if ``arr`` is an ``xarray.DataArray`` then ``time`` can be either
          a name of the time dimension or ``None`` - in the latter case the
          time coordinates are taken from ``'time'`` coordinate of ``arr``.
        * if ``arr`` is an ``mne.Epochs`` object then ``time`` is ignored.
    sfreq : float | None
        Sampling frequency. Inferred from analog signal time dimension if not
        provided (or taken from ``.info['sfreq']`` if ``arr`` is ``mne.Epochs``
        ).
    winlen : float
        Window length in seconds. This is the full window length: a window
        length of ``0.1`` means that the window will start 0.05 seconds before
        and end 0.05 seconds after the spike time. Defaults to ``0.1``.

    Returns
    -------
    spike_centered : xarray.DataArray
        Spike-centered windows. ``n_spikes x n_channels x n_times``.
    '''
    import xarray as xr
    from borsar.utils import find_index

    # check inputs
    picks = _deal_with_picks(spk, pick)
    cell_idx = picks[0]
    ch_names = None
    unit = None
    metadata = spk.metadata

    if isinstance(arr, xr.DataArray):
        if time is None:
            if 'time' not in arr.coords:
                raise ValueError('When ``time=None`` the ``arr`` xarray has '
                                 'to contain a coordinate named "time". '
                                 'Alternatively, pass the name of the time '
                                 'coordinate in ``time`` input argument.')
            time = arr.coords['time'].values
        else:
            if isinstance(time, str):
                if time not in arr.coords:
                    raise ValueError(f'Coordinate named "{time}" not found in '
                                     '``arr`` xarray.')
                time = arr.coords[time].values
            else:
                raise ValueError('When ``arr`` is an xarray ``time`` input '
                                 'argument has to be either ``None`` or a '
                                 f'string, got {type(time)}.')

    elif isinstance(arr, np.ndarray):
        if time is None:
            raise ValueError('When ``arr`` is an ndarray ``time`` input '
                                'argument has to be an array of time points '
                                'for each sample of the time dimension.')
    else:
        import mne
        if isinstance(arr, mne.Epochs):
            time = arr.times
            ch_names = arr.ch_names
            sfreq = arr.info['sfreq']
            unit = 'V'
            if metadata is None:
                metadata = arr.metadata

            arr = arr.get_data()
        else:
            raise ValueError('``arr`` has to be either an xarray, numpy array '
                             f'or mne.Epochs, got {type(arr)}.')

    if sfreq is None:
        sfreq = 1 / np.diff(time).mean()
    if ch_names is None:
        n_channels = arr.shape[1]
        ch_names = np.arange(n_channels)

    spike_centered = list()
    window_samples, half_win = _symmetric_window_samples(winlen, sfreq)
    window_time = window_samples / sfreq
    winlims = np.array([-half_win, half_win + 1])[None, :]
    lims = [0, len(time)]
    tri_is_ok = np.zeros(len(spk.trial[cell_idx]), dtype='bool')

    n_tri = max(spk.trial[cell_idx]) + 1
    for tri_idx in range(n_tri):
        # ENH: change to trial boundaries
        sel = spk.trial[cell_idx] == tri_idx
        if sel.any():
            tms = spk.time[cell_idx][sel]

            closest_smp = find_index(time, tms)
            twins = closest_smp[:, None] + winlims
            good = ((twins >= lims[0]) & (twins <= lims[1])).all(axis=1)
            twins = twins[good]
            tri_is_ok[sel] = good

            for twin in twins:
                sig_part = arr[tri_idx, :, twin[0]:twin[1]]
                spike_centered.append(sig_part)

    # stack windows
    spike_centered = np.stack(spike_centered, axis=0)

    # prepare coordinates
    spike_idx = np.where(tri_is_ok)[0]
    tri = spk.trial[cell_idx][tri_is_ok]

    coords = {'spike': spike_idx, 'channel': ch_names,
              'time': window_time, 'trial': ('spike', tri)}
    coords = _inherit_metadata(coords, metadata, 'spike', tri=tri)

    # construct xarray and assign coords
    spike_centered = xr.DataArray(
        spike_centered, dims=['spike', 'channel', 'time'],
        coords=coords, name='amplitude')
    if unit is not None:
        spike_centered.attrs['unit'] = unit

    return spike_centered


# TODO: differentiate between shuffling spike-trials vs just metadata
def shuffle_trials(spk, drop_timestamps=True, drop_waveforms=True):
    '''Create a copy of the SpikeEpochs object with shuffled trials.

    Parameters
    ----------
    spk : SpikeEpochs
        SpikeEpochs object.
    drop_timestamps : bool
        If True, timestamps are not copied to the new object.
    drop_waveforms : bool
        If True, waveforms are not copied to the new object.

    Returns
    -------
    new_spk : SpikeEpochs
        SpikeEpochs object with shuffled trials.
    '''
    new_spk = spk.copy()

    n_tri = spk.n_trials
    n_cells = spk.n_units()
    tri_idx = np.arange(n_tri)
    np.random.shuffle(tri_idx)

    has_timestamps = spk.timestamps is not None
    has_waveforms = spk.waveform is not None

    if drop_timestamps:
        new_spk.timestamps = None
        has_timestamps = False
    if drop_waveforms:
        new_spk.waveform = None
        has_waveforms = False

    for cell_idx in range(n_cells):
        tri_limits, tri_id = _get_trial_boundaries(spk, cell_idx)
        n_spikes = np.diff(tri_limits)

        start_idx = 0
        for ix, tri in enumerate(tri_idx):
            pos = np.where(tri_id == tri)[0]
            if len(pos) > 0:
                pos = pos[0]
                limits = tri_limits[pos:pos + 2]
                n_spk = n_spikes[pos]

                slc = slice(start_idx, start_idx + n_spk)
                new_spk.trial[cell_idx][slc] = ix
                new_spk.time[cell_idx][slc] = (
                    spk.time[cell_idx][limits[0]:limits[1]]
                )

                if has_timestamps:
                    new_spk.timestamps[cell_idx][slc] = (
                        spk.timestamps[cell_idx][limits[0]:limits[1]]
                    )

                if has_waveforms:
                    new_spk.waveform[cell_idx][slc] = (
                        spk.waveform[cell_idx][limits[0]:limits[1]]
                    )

                start_idx += n_spk
    return new_spk


# TODO - if other sorters are used, alignment point (sample_idx) for the
#        spike waveforms should be saved somewhere in spk and used here.
def infer_waveform_polarity(spk, cell_idx, threshold=1.75, baseline_range=50,
                            rich_output=False):
    """Decide whether waveform polarity is positive, negative or unknown.

    This may be useful to detect units/clusters with bad alignment.
    The decision is based on comparing baselined min and max average waveform
    peak values. The value for the peak away from alignment point is calculated
    from single spike waveforms to simulate alignment and reduce bias (30
    samples around that peak are taken and min/max values for this time window).
    The alignment point is expected where it is for osort - around sample 92.

    Parameters
    ----------
    spk : pylabianca.Spikes
        Spikes object to use.
    cell_idx : int
        Index of the cell whose waveform should be checked.
    threshold : float
        Threshold ratio for the minimum and maximum waveform peak values to
        decide about polarity. Default is ``1.75``, which means that one of
        the peaks (min or max) must be at least 1.75 times higher than the
        other to decide on polarity. If given waveform does not pass this
        test it is labelled as ``'unknown'``.
    baseline_range : int
        Number of first samples to use as baseline. Default is ``50``.
    rich_output : bool
        If True, return a dictionary with the following fields:
        * 'type' : 'positive' or 'negative' or 'unknown'
        * 'min_peak' : minimum peak value
        * 'max_peak' : maximum peak value
        * 'min_idx' : index of the minimum peak
        * 'max_idx' : index of the maximum peak
        * 'align_idx' : index of the alignment point
        * 'align_sign' : polarity of the alignment point (-1 or 1)

    Returns
    -------
    unit_type : str | dict
        Polarity label for the waveform. Either ``'positive'``, ``'negative'``
        or ``'unknown'``. If ``rich_output`` is True, a dictionary with
        multiple fields is returned (see description of ``rich_output``
        argument).
    """

    inv_threshold = 1 / threshold

    # decide whether the waveform is pos or neg
    avg_waveform = spk.waveform[cell_idx].mean(axis=0)
    min_val_idx, max_val_idx = avg_waveform.argmin(), avg_waveform.argmax()
    min_val, max_val = avg_waveform.min(), avg_waveform.max()

    # the value not aligned to will be underestimated, correct for that ...
    further_away = np.abs(np.array([min_val_idx, max_val_idx]) - 92).argmax()
    operation = [np.min, np.max][further_away]
    away_idx = [min_val_idx, max_val_idx][further_away]

    # ... by estimating this value in a wider window
    rng = slice(away_idx - 15, away_idx + 15)
    slc = spk.waveform[cell_idx][:, rng]
    this_val = operation(slc, axis=1).mean()

    if further_away == 0:
        min_val = this_val
    else:
        max_val = this_val

    # the min and max values are baselined to further reduce bias
    baseline = avg_waveform[:baseline_range].mean()
    min_val -= baseline
    max_val -= baseline

    # based on min / max ratio a decision is made
    prop = min_val / max_val

    if np.abs(prop) > threshold:
        unit_type = 'neg'
    elif np.abs(prop) < inv_threshold:
        unit_type = 'pos'
    else:
        unit_type = 'unknown'

    if not rich_output:
        return unit_type
    else:
        align_which = 1 - further_away
        align_idx = [min_val_idx, max_val_idx][align_which]
        align_sign = [-1, 1][align_which]
        output = {'type': unit_type, 'min_peak': min_val, 'max_peak': max_val,
                  'min_idx': min_val_idx, 'max_idx': max_val_idx,
                  'align_idx': align_idx, 'align_sign': align_sign}
        return output


def _realign_waveforms(waveforms, pad_nans=False, reject=True):
    '''Realign waveforms. Used in ``realign_waveforms()`` function.'''
    mean_wv = np.nanmean(waveforms, axis=0)
    min_idx, max_idx = np.argmin(mean_wv), np.argmax(mean_wv)

    if min_idx < max_idx:
        waveforms *= -1
        mean_wv *= -1
        min_idx, max_idx = max_idx, min_idx

    # checking slope
    # --------------
    if reject:
        slope = np.nansum(np.diff(waveforms[:, :max_idx], axis=1), axis=1)
        bad_slope = slope < 0

    # realigning
    # ----------
    spike_max = np.argmax(waveforms, axis=1)
    new_waveforms = np.empty(waveforms.shape)
    new_waveforms.fill(np.nan)

    unique_mx = np.unique(spike_max)

    if reject:
        n_samples = waveforms.shape[1]
        max_dist = int(n_samples / 5)
        dist_to_peak = np.abs(spike_max - max_idx)
        bad_peak_dist = dist_to_peak > max_dist

        unique_mx = unique_mx[np.abs(unique_mx - max_idx) <= max_dist]

    for uni_ix in unique_mx:
        diff_idx = max_idx - uni_ix
        spk_msk = spike_max == uni_ix

        if diff_idx == 0:
            new_waveforms[spk_msk, :] = waveforms[spk_msk, :]
        elif diff_idx > 0:
            # individual peak too early
            new_waveforms[spk_msk, diff_idx:] = waveforms[spk_msk, :-diff_idx]

            if not pad_nans:
                new_waveforms[spk_msk, :diff_idx] = (
                    waveforms[spk_msk, [0]][:, None])
        else:
            # individual peak too late
            new_waveforms[spk_msk, :diff_idx] = waveforms[spk_msk, -diff_idx:]

            if not pad_nans:
                new_waveforms[spk_msk, diff_idx:] = (
                    waveforms[spk_msk, [diff_idx - 1]][:, None])

    waveforms_to_reject = (np.where(bad_slope | bad_peak_dist)[0]
                           if reject else None)

    return new_waveforms, waveforms_to_reject


def realign_waveforms(spk, picks=None, min_spikes=10, reject=True):
    '''Realign single waveforms compared to average waveform. Works in place.

    Parameters
    ----------
    spk :  pylabianca.Spikes | pylabianca.SpikeEpochs
        Spikes or SpikeEpochs object.
    picks : int | str | list-like of int | list-like of str
        The units to realign waveforms for.
    min_spikes : int
        Minimum number of spikes to try realigning the waveform.
    reject : bool
        Also remove waveforms and
    '''
    picks = _deal_with_picks(spk, picks)
    for cell_idx in picks:
        waveforms = spk.waveform[cell_idx]
        if waveforms is not None and len(waveforms) > min_spikes:
            waveforms, reject_idx = _realign_waveforms(waveforms)
            spk.waveform[cell_idx] = waveforms

            # reject spikes
            # TODO: could be made a separate function one day
            n_reject = len(reject_idx)
            if n_reject > 0:
                msg = (f'Removing {n_reject} bad waveforms for cell'
                       f'{spk.cell_names[cell_idx]}.')
                print(msg)

                spk.waveform[cell_idx] = np.delete(
                    spk.waveform[cell_idx], reject_idx, axis=0)
                spk.timestamps[cell_idx] = np.delete(
                    spk.timestamps[cell_idx], reject_idx)


def _get_trial_boundaries(spk, cell_idx):
    n_spikes = len(spk.trial[cell_idx])
    trial_boundaries = np.where(np.diff(spk.trial[cell_idx]))[0] + 1
    trial_boundaries = np.concatenate(
        [[0], trial_boundaries, [n_spikes]])
    tri_num = spk.trial[cell_idx][trial_boundaries[:-1]]

    return trial_boundaries, tri_num


# TODO - this can be made more universal
def find_cells_by_cluster_id(spk, cluster_ids, channel=None):
    '''Find cell indices that create given clusters on specific channel.'''
    cell_idx = list()
    if isinstance(cluster_ids, int):
        cluster_ids = [cluster_ids]

    for cl in cluster_ids:
        is_cluster = spk.cellinfo.cluster == cl
        if channel is not None:
            is_channel = spk.cellinfo.channel == channel
            idxs = np.where(is_cluster & is_channel)[0]
        else:
            idxs = np.where(is_cluster)[0]

        if len(idxs) == 1:
            cell_idx.append(idxs[0])
        else:
            raise ValueError('Found 0 or > 1 cluster IDs.')

    return cell_idx


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


def drop_cells_by_channel_and_cluster_id(spk, to_drop):
    '''Works in place!'''
    # find cell idx by channel + cluster ID
    cell_idx = list()
    for channel, cluster in to_drop:
        this_idx = find_cells_by_cluster_id(spk, cluster, channel=channel)[0]
        cell_idx.append(this_idx)
    spk.drop_cells(cell_idx)


def get_data_path():
    home_dir = os.path.expanduser('~')
    data_dir = 'pylabianca_data'
    full_data_dir = op.join(home_dir, data_dir)
    has_data_dir = op.exists(full_data_dir)

    if not has_data_dir:
        os.mkdir(full_data_dir)

    return full_data_dir


def get_fieldtrip_data():
    import pooch

    data_path = get_data_path()
    ft_url = ('https://download.fieldtriptoolbox.org/tutorial/spike/p029_'
              'sort_final_01.nex')
    known_hash = ('4ae4ed2a9613cde884b62d8c5713c418cff5f4a57c8968a3886'
                  'db1e9991a81c9')
    fname = pooch.retrieve(
        url=ft_url, known_hash=known_hash,
        fname='p029_sort_final_01.nex', path=data_path
    )
    return fname


def get_test_data_link():
    dropbox_lnk = ('https://www.dropbox.com/scl/fo/757tf3ujqga3sa2qocm4l/h?'
                   'rlkey=mlz44bcqtg4ds3gsc29b2k62x&dl=1')
    return dropbox_lnk


def download_test_data():
    # check if test data exist
    data_dir = get_data_path()
    check_files = [
        'ft_spk_epoched.mat', 'monkey_stim.csv', ('p029_sort_final_01_events'
        '.mat'), r'test_osort_data\sub-U04_switchorder\CSCA130_mm_format.mat',
        r'test_neuralynx\sub-U06_ses-screening_set-U6d_run-01_ieeg\CSC129.ncs'
    ]

    if all([op.isfile(op.join(data_dir, f)) for f in check_files]):
        return

    import pooch
    import zipfile

    # set up paths
    fname = 'temp_file.zip'
    download_link = get_test_data_link()

    # download the file
    hash = None
    pooch.retrieve(url=download_link, known_hash=hash,
                   path=data_dir, fname=fname)

    # unzip and extract
    # TODO - optionally extract only the missing files
    destination = op.join(data_dir, fname)
    zip_ref = zipfile.ZipFile(destination, 'r')
    zip_ref.extractall(data_dir)
    zip_ref.close()

    # remove the zipfile
    os.remove(destination)


def has_elephant():
    '''Test if elephant is available.'''
    try:
        import elephant
        return True
    except ImportError:
        return False


def create_random_spikes(n_cells=4, n_trials=25, n_spikes=(10, 21),
                         **args):
    '''Create random spike data. Mostly useful for testing.

    Parameters
    ----------
    n_cells : int
        Number of cells.
    n_trials : int
        Number of trials. If ``None`` or 0 then Spikes object is returned.
    n_spikes : int | tuple
        Number of spikes. If tuple then the first element is the minimum
        number of spikes and the second element is the maximum number of
        spikes.
    args : dict
        Additional arguments are passed to the Spikes / SpikeEpochs object.

    Returns
    -------
    spikes : Spikes | SpikeEpochs
        Spike data object.
    '''
    from .spikes import SpikeEpochs, Spikes

    tmin, tmax = -0.5, 1.5
    tlen = tmax - tmin
    constant_n_spikes = isinstance(n_spikes, int)
    if constant_n_spikes:
        n_spk = n_spikes

    return_epochs = isinstance(n_trials, int) and n_trials > 0
    if not return_epochs:
        n_trials = 1
        tmin = 0
        tmax = 1e6

    times = list()
    trials = list()
    for _ in range(n_cells):
        this_tri = list()
        this_tim = list()
        for tri_idx in range(n_trials):
            if not constant_n_spikes:
                n_spk = np.random.randint(*n_spikes)

            if return_epochs:
                tms = np.random.rand(n_spk) * tlen + tmin
                this_tri.append(np.ones(n_spk, dtype=int) * tri_idx)
            else:
                tms = np.random.randint(tmin, tmax, size=n_spk)
            tms = np.sort(tms)
            this_tim.append(tms)

        this_tim = np.concatenate(this_tim)
        times.append(this_tim)

        if return_epochs:
            this_tri = np.concatenate(this_tri)
            trials.append(this_tri)

    if return_epochs:
        return SpikeEpochs(times, trials, **args)
    else:
        if 'sfreq' not in args:
            args['sfreq'] = 10_000

        return Spikes(times, **args)


def is_list_or_object_array(obj):
    return (isinstance(obj, list)
            or (isinstance(obj, np.ndarray)
                and np.issubdtype(obj.dtype, np.object_))
    )


def _validate_spike_epochs_input(time, trial):
    '''Validate input for SpikeEpochs object.'''

    # both time and trial have to be lists ...

    if not (is_list_or_object_array(time) and is_list_or_object_array(trial)):
        raise ValueError('Both time and trial have to be lists or object '
                         'arrays.')

    # ... of the same length ...
    if len(time) != len(trial):
        raise ValueError('Length of time and trial lists must be the same.')

    # ... and all elements have to be numpy arrays
    if not all([isinstance(cell_time, np.ndarray) for cell_time in time]):
        raise ValueError('All elements of time list must be numpy arrays.')
    if not all([isinstance(cell_trial, np.ndarray) for cell_trial in trial]):
        raise ValueError('All elements of trial list must be numpy arrays.')

    # all corresponding time and trial arrays have to have the same length
    if not all([len(time[ix]) == len(trial[ix]) for ix in range(len(time))]):
        raise ValueError('All time and trial arrays must have the same length.')

    # trial arrays have to contain non-negative integers
    for cell_trial in trial:
        if not (np.issubdtype(cell_trial.dtype, np.integer)
                and cell_trial.min() >= 0):
            raise ValueError(
                'Trial list of arrays must contain non-negative integers.')


def _validate_spikes_input(times):
    '''Validate input for SpikeEpochs object.'''

    # both time and trial have to be lists ...

    if not is_list_or_object_array(times):
        raise ValueError('Timestamps have to be lists or object arrays.')

    # ... and all elements have to be numpy arrays
    if not all([isinstance(cell_times, np.ndarray) for cell_times in times]):
        raise ValueError('All elements of timestamp list must be numpy '
                         'arrays.')

    # timestamp arrays have to contain non-negative integers
    for cell_times in times:
        if not (np.issubdtype(cell_times.dtype, np.integer)
                and cell_times.min() >= 0):
            raise ValueError(
                'Timestamp lists of arrays must contain non-negative '
                'integers.')


def _validate_cellinfo(spk, cellinfo):
    '''Validate cellinfo input for SpikeEpochs object.'''
    if cellinfo is not None:
        if not isinstance(cellinfo, pd.DataFrame):
            raise ValueError('cellinfo has to be a pandas DataFrame.')
        if cellinfo.shape[0] != spk.n_units():
            raise ValueError('Number of rows in cellinfo has to be equal to '
                             'the number of cells in the SpikeEpochs object.')
        if not (cellinfo.index == np.arange(spk.n_units())).all():
            warn('cellinfo index does not match cell indices in the '
                 'SpikeEpochs object. Resetting the index.')
            cellinfo = cellinfo.reset_index(drop=True)

    return cellinfo


def xr_find_nested_dims(arr, dim_name):
    names = list()
    coords = list(arr.coords)
    coords.remove(dim_name)
    sub_dim = (dim_name,)
    for coord in coords:
        if arr.coords[coord].dims == sub_dim:
            names.append(coord)

    return names
