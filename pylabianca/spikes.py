import numpy as np
import pandas as pd

from .utils import _deal_with_picks, _turn_spike_rate_to_xarray


# TODO:
# - [ ] index by trial?
# - [ ] maybe passing `n_trials` does not make so much sense? If it is not used
#       in other places - then maybe not.
class SpikeEpochs():
    def __init__(self, time, trial, time_limits=None, n_trials=None,
                 cell_names=None, metadata=None, cellinfo=None):
        '''Create ``SpikeEpochs`` object for convenient storage, analysis and
        visualization of spikes data.

        Parameters
        ----------
        time : listlike of np.ndarray
            List of arrays, where each array contains spike times for one
            cell (neuron). The times are centered with respect to epoch onset
            (for example stimulus presentation).
        trial : listlike of np.ndarray
            List of arrays, where each array contains trial membership of
            spikes from respective ``time`` array for one cell (neuron). The
            trial indices are zero-based integers.
        time_limits : listlike | None
            Optional. Two-element array with epoch time limits with respect to
            epoch-centering event. The limits have to be in seconds.
            For example ``np.array([-0.5, 1.5])`` means from 0.5 seconds
            before the event up to 1.5 seconds after the event. The default
            (``None``) infers time limits from min and max spike times.
        n_trials : int | None
            Number of trials. Optional, if the number of trials can't be
            inferred from the ``trials`` argument (for example when none of the
            cells fire for the last few trials).
        cell_names : list of str | None
            String identifiers of cells. First string corresponds to first
            cell, that is ``time[0]`` and ``trial[0]`` (and so forth).
            Optional, the default (``None``) names the first cell
            ``'cell000'``, the second cell ``'cell001'`` and so on.
        metadata : pandas.DataFrame
            DataFrame with trial-level metadata.
        cellinfo : pandas.DataFrame
            DataFrame with additional information about the cells.
        '''
        if not isinstance(time[0], np.ndarray):
            time = [np.asarray(x) for x in time]
        if not isinstance(trial[0], np.ndarray):
            trial = [np.asarray(x) for x in trial]
        self.time = time
        self.trial = trial

        if time_limits is None:
            tmin = min([min(x) for x in time])
            tmax = max([max(x) for x in time])
            time_limits = np.array([tmin, tmax])
        self.time_limits = time_limits

        if n_trials is None:
            n_trials = max(max(tri) + 1 for tri in self.trial)
        if cell_names is None:
            n_cells = len(time)
            cell_names = np.array(['cell{:03d}'.format(idx)
                                   for idx in range(n_cells)])
        else:
            cell_names = np.asarray(cell_names)

        if metadata is not None:
            assert isinstance(metadata, pd.DataFrame)
            assert metadata.shape[0] == n_trials

        n_cells = len(self.time)
        if cellinfo is not None:
            assert isinstance(cellinfo, pd.DataFrame)
            assert cellinfo.shape[0] == n_cells

        self.n_trials = n_trials
        self.cell_names = cell_names
        self.metadata = metadata
        self.cellinfo = cellinfo

    def __repr__(self):
        '''Text representation of SpikeEpochs.'''
        n_cells = len(self.time)
        avg_spikes = np.mean([len(x) for x in self.time])
        msg = '<SpikeEpochs, {} epochs, {} cells, {:.1f} spikes/cell on average>'
        return msg.format(self.n_trials, n_cells, avg_spikes)

# TODO: ability to get a shallow copy might also be useful
    def copy(self):
        '''Return a deep copy of the object.'''
        from copy import deepcopy
        return deepcopy(self)

    def pick_cells(self, picks=None, query=None):
        '''Select cells by name or index. Operates inplace.'''
        if picks is None and query is None:
            return self

        if picks is None and query is not None:
            assert self.cellinfo is not None
            cellinfo_sel = self.cellinfo.query(query)
            picks = cellinfo_sel.index.values
        else:
            picks = _deal_with_picks(self, picks)

        self.time = [self.time[ix] for ix in picks]
        self.trial = [self.trial[ix] for ix in picks]
        self.cell_names = self.cell_names[picks].copy()
        if self.cellinfo is not None:
            self.cellinfo = self.cellinfo.loc[picks, :].reset_index(drop=True)

        return self

    def crop(self, tmin=None, tmax=None):
        '''Confine time range to specified limit. Operates inplace.

        Parameters
        ----------
        tmin : float | None
            Lower time bound in seconds. Spikes later or equal (``>=``) to this
            time will be retained.
        tmax : float | None
            Higher time bound in seconds. Spikes earlier or equal (``<=``) to
            this time will be retained.
        '''
        if tmin is None and tmax is None:
            raise TypeError('You have to specify tmin and/or tmax.')

        if tmin is None:
            tmin = self.time_limits[0]
        if tmax is None:
            tmax = self.time_limits[1]

        for cell_idx in range(len(self.time)):
            sel = (self.time[cell_idx] >= tmin) & (self.time[cell_idx] <= tmax)
            self.time[cell_idx] = self.time[cell_idx][sel]
            self.trial[cell_idx] = self.trial[cell_idx][sel]
        self.time_limits = [tmin, tmax]
        return self

    # TODO: refactor (DRY: merge both loops into one?)
    # TODO: better handling of numpy vs numba implementation
    # TODO: consider adding `return_type` with `Epochs` option (mne object)
    def spike_rate(self, picks=None, winlen=0.25, step=0.01, tmin=None,
                   tmax=None, backend='numpy'):
        '''Calculate spike rate with a running window.

        Parameters
        ----------
        picks : int | listlike of int | None
            The neuron index to use in the calculations. The default (``None``)
            uses all cells.
        winlen : float
            Length of the running window in seconds.
        step : float | bool
            The step size of the running window. If step is ``False`` then
            spike rate is not calculated using a running window but with
            a static one with limits defined by ``tmin`` and ``tmax``.
        backend : str
            Can be ``'numpy'`` or ``'numba'``.

        Returns
        -------
        frate : xarray.DataArray
            Xarray with following labeled dimensions: cell, trial, time.
        '''
        picks = _deal_with_picks(self, picks)

        if isinstance(step, bool) and not step:
            assert tmin is not None
            assert tmax is not None
            times = f'{tmin} - {tmax} s'

            frate = list()
            cell_names = list()

            for pick in picks:
                frt = _compute_spike_rate_fixed(
                    self.time[pick], self.trial[pick], [tmin, tmax],
                    self.n_trials)
                frate.append(frt)
                cell_names.append(self.cell_names[pick])

        else:
            frate = list()
            cell_names = list()

            if backend == 'numpy':
                func = _compute_spike_rate_numpy
            elif backend == 'numba':
                from ._numba import _compute_spike_rate_numba
                func = _compute_spike_rate_numba
            else:
                raise ValueError('Backend can be only "numpy" or "numba".')

            for pick in picks:
                times, frt = func(
                    self.time[pick], self.trial[pick], self.time_limits,
                    self.n_trials, winlen=winlen, step=step)
                frate.append(frt)
                cell_names.append(self.cell_names[pick])

        frate = np.stack(frate, axis=0)
        frate = _turn_spike_rate_to_xarray(times, frate, self,
                                           cell_names=cell_names)
        return frate

    def spike_density(self, picks=None, winlen=0.3, gauss_sd=None, sfreq=500.):
        '''Compute spike density by convolving spikes with a gaussian kernel.

        Parameters
        ----------
        picks : int | listlike of int | None
            The neuron indices to use in the calculations. The default
            (``None``) uses all cells.
        winlen : float
            Length of the gaussian kernel window. By default, if
            ``gauss_sd=None``, the standard deviation of the kernel is
            ``winlen / 6``.
        gauss_sd : float | None
            Standard deviation of the gaussian kernel.
        sfreq : float
            Desired sampling frequency of the spike density. Defaults to 500
            Hz.

        Returns
        -------
        frate : xarray.DataArray
            Spike density in a cell x trial x time xarray.
        '''
        picks = _deal_with_picks(self, picks)
        tms, cnt = _spike_density(self, picks=picks, winlen=winlen,
                                  gauss_sd=gauss_sd, sfreq=sfreq)
        xarr = _turn_spike_rate_to_xarray(
            tms, cnt.transpose((1, 0, 2)), self,
            cell_names=self.cell_names[picks])
        return xarr

    # TODO: empty trials are dropped by default now...
    # - [ ] use `group` from sarna in looping through trials
    #       for faster execution...
    def to_neo(self, cell_idx, pool=False):
        '''Turn spikes of given cell into neo.SpikeTrain format.

        Parameters
        ----------
        cell_idx : int
            Index of the cell to turn into neo.SpikeTrain format.
        pool : bool
            Whether to pool all the trials into a single neo.SpikeTrain.

        Returns
        -------
        spikes : neo.SpikeTrain | list of neo.SpikeTrain
            If ``pool=True``, a single neo.SpikeTrain is returned with spikes
            from separate trials pooled and sorted. Otherwise (``pool=False``)
            a list of neo.SpikeTrain objects is returned.
        '''
        import neo
        from quantities import s

        if not pool:
            spikes = list()
            trials = range(self.n_trials)
            for tri in trials:
                times = self.time[cell_idx][self.trial[cell_idx] == tri]
                spiketrain = neo.SpikeTrain(
                    times * s, t_stop=self.time_limits[1],
                    t_start=self.time_limits[0])
                spikes.append(spiketrain)
        else:
            times = np.sort(self.time[cell_idx])
            spikes = neo.SpikeTrain(
                times * s, t_stop=self.time_limits[1],
                t_start=self.time_limits[0])
        return spikes


# TODO: the implementation and the API are suboptimal
def compare_spike_times(spk, cell_idx1, cell_idx2, tol=0.002):
    '''Test concurrence of spike times for SpikeEpochs.

    Parameters
    ----------
    spk : SpikeEpochs
        SpikeEpochs object.
    cell_idx1 : int
        Index of the first cell to compare. All spikes of this cell will be
        checked.
    cell_idx2 : int
        Index of the second cell to compare. Spikes of the first cell will
        be tested for concurrence with spikes coming from this cell.
    tol : float
        Concurrence tolerance in seconds. Spikes no further that this will be
        deemed coocurring.

    Returns
    -------
    float
        Percentage of spikes from first cell that concur with spikes from the
        second cell.
    '''
    if isinstance(spk, SpikeEpochs):
        tri1, tms1 = spk.trial[cell_idx1], spk.time[cell_idx1]
        tri2, tms2 = spk.trial[cell_idx2], spk.time[cell_idx2]

        n_spikes = len(tri1)
        if_match = np.zeros(n_spikes, dtype='bool')
        for idx in range(n_spikes):
            this_time = tms1[idx]
            this_tri = tri1[idx]
            corresp_tri = np.where(tri2 == this_tri)[0]
            match = False
            if len(corresp_tri) > 0:
                corresp_tm = tms2[corresp_tri]
                match = (np.abs(corresp_tm - this_time) < tol).any()
            if_match[idx] = match
        return if_match.mean()
    elif isinstance(spk, Spikes):
        tms1 = spk.timestamps[cell_idx1] / spk.sfreq
        tms2 = spk.timestamps[cell_idx2] / spk.sfreq
        time_diffs = np.abs(tms1[:, None] - tms2[None, :])
        closest_time1 = time_diffs.min(axis=1)
        return (closest_time1 < tol).mean()


def _epoch_spikes(timestamps, event_times, tmin, tmax):
    '''Epoch spike data with respect to event timestamps.

    Helper function that epochs spikes for a single neuron.

    Parameters
    ----------
    timestamps : numpy array
        Array containing spike timestamps.
    event_times : numpy array
        Array containing event timestamps.
    tmin : float
        Lower epoch limit.
    tmax : float
        Upper epoch limit.

    Returns
    -------
    trial : numpy array
        Information about the trial that the given spike belongs to.
    time : numpy array
        Spike times with respect to event onset.
    '''
    trial = list()
    time = list()

    t_idx = 0
    n_epochs = event_times.shape[0]
    this_epo = (timestamps[t_idx] < (event_times + tmax)).argmax()

    for epo_idx in range(this_epo, n_epochs):
        # find spikes that fit within the epoch
        first_idx = (timestamps[t_idx:] > (
            event_times[epo_idx] + tmin)).argmax() + t_idx
        msk = timestamps[first_idx:] < (event_times[epo_idx] + tmax)

        # select these spikes and center wrt event time
        tms = timestamps[first_idx:][msk] - event_times[epo_idx]
        if len(tms) > 0:
            tri = np.ones(len(tms), dtype='int') * epo_idx
            trial.append(tri)
            time.append(tms)
        t_idx = first_idx

    if len(trial) > 0:
        trial = np.concatenate(trial)
        time = np.concatenate(time)
    else:
        trial = [np.array([])]
        time = [np.array([])]
    return trial, time


# TODO: make this a method of SpikeEpochs and return xarray or mne.Epochs
def _spikes_to_raw(spk, picks=None, sfreq=500.):
    '''Turn epoched spike timestamps into binary representation.

    Parameters
    ----------
    spk : SpikeEpochs
        Spikes to turn into binary representation.
    picks : list-like of int | None
        Cell indices to turn to binary representation. Optional, the default
        (``None``) uses all available neurons.
    sfreq : float
        Sampling frequency of the the output array. Defaults to ``500.``.

    Returns
    -------
    times : numpy array
        1d array of time labels.
    trials_raw : numpy array
        ``trials x cells x timesamples`` array with binary spike information.
    '''
    picks = _deal_with_picks(spk, picks)
    sample_time = 1 / sfreq
    tmin, tmax = spk.time_limits
    times = np.arange(tmin, tmax + 0.01 * sample_time, step=sample_time)

    n_cells = len(picks)
    n_times = len(times)
    trials_raw = np.zeros((spk.n_trials, n_cells, n_times), dtype='int')

    for idx, cell_idx in enumerate(picks):
        from_idx = 0
        for this_tri in range(spk.n_trials):
            ix = np.where(spk.trial[cell_idx][from_idx:] == this_tri)[0]
            if len(ix) == 0:
                continue

            ix = ix[-1] + 1
            spike_times = spk.time[cell_idx][from_idx:from_idx + ix]
            sample_ix = (np.abs(times[:, np.newaxis]
                                - spike_times[np.newaxis, :]).argmin(axis=0))
            t_smp, n_spikes = np.unique(sample_ix, return_counts=True)
            trials_raw[this_tri, idx, t_smp] = n_spikes

            from_idx = from_idx + ix
    return times, trials_raw


def _compute_spike_rate_numpy(spike_times, spike_trials, time_limits,
                              n_trials, winlen=0.25, step=0.05):
    halfwin = winlen / 2
    epoch_len = time_limits[1] - time_limits[0]
    n_steps = int(np.floor((epoch_len - winlen) / step + 1))

    fr_t_start = time_limits[0] + halfwin
    fr_tend = time_limits[1] - halfwin + step * 0.001
    times = np.arange(fr_t_start, fr_tend, step=step)
    frate = np.zeros((n_trials, n_steps))

    for step_idx in range(n_steps):
        winlims = times[step_idx] + np.array([-halfwin, halfwin])
        msk = (spike_times >= winlims[0]) & (spike_times < winlims[1])
        tri = spike_trials[msk]
        in_tri, count = np.unique(tri, return_counts=True)
        frate[in_tri, step_idx] = count / winlen

    return times, frate


# @numba.jit(nopython=True)
# currently raises warnings, and jit is likely not necessary here
# Encountered the use of a type that is scheduled for deprecation: type
# 'reflected list' found for argument 'time_limits' of function
# '_compute_spike_rate_fixed'
def _compute_spike_rate_fixed(spike_times, spike_trials, time_limits,
                              n_trials):

    winlen = time_limits[1] - time_limits[0]
    frate = np.zeros(n_trials)
    msk = (spike_times >= time_limits[0]) & (spike_times < time_limits[1])
    tri = spike_trials[msk]
    intri, count = np.unique(tri, return_counts=True)
    frate[intri] = count / winlen

    return frate


def _symmetric_window_samples(winlen, sfreq):
    half_len_smp = int(np.round(winlen / 2 * sfreq))
    win_smp = np.arange(-half_len_smp, half_len_smp + 1)
    return win_smp, half_len_smp


def _gauss_kernel_samples(window, gauss_sd):
    from scipy.stats.distributions import norm
    kernel = norm(loc=0, scale=gauss_sd)
    kernel = kernel.pdf(window)
    return kernel


# TODO: consider an exact mode where the spikes are not transformed to raw
#       but placed exactly where the spike is (`loc=spike_time`) and evaluated
#       (maybe this is what is done by elephant?)
# TODO: check if time is symmetric wrt 0 (in most cases it should be as epochs
#       are constructed wrt specific event)
def _spike_density(spk, picks=None, winlen=0.3, gauss_sd=None, kernel=None,
                   sfreq=500.):
    '''Calculates normal (constant) spike density.

    The density is computed by convolving the binary spike representation
    with a gaussian kernel.
    '''
    from scipy.signal import correlate

    if kernel is None:
        gauss_sd = winlen / 6 if gauss_sd is None else gauss_sd
        gauss_sd = gauss_sd * sfreq

        win_smp, trim = _symmetric_window_samples(winlen, sfreq)
        kernel = _gauss_kernel_samples(win_smp, gauss_sd) * sfreq
    else:
        assert (len(kernel) % 2) == 1
        trim = int((len(kernel) - 1) / 2)

    picks = _deal_with_picks(spk, picks)
    times, binrep = _spikes_to_raw(spk, picks=picks, sfreq=sfreq)
    cnt_times = times[trim:-trim]

    cnt = correlate(binrep, kernel[None, None, :], mode='valid')
    return cnt_times, cnt


def depth_of_selectivity(frate, by):
    '''Compute depth of selectivity for given category.

    Parameters
    ----------
    frate : xarray
        Xarray with firing rate data.
    by : str
        Name of the dimension to group by and calculate depth of
        selectivity for.

    Returns
    -------
    selectivity : xarray
        Xarray with depth of selectivity.
    '''
    avg_by_probe = frate.groupby(by).mean(dim='trial')
    n_categories = len(avg_by_probe.coords[by])
    r_max = avg_by_probe.max(by)
    numerator = n_categories - (avg_by_probe / r_max).sum(by)
    selectivity = numerator / (n_categories - 1)
    return selectivity, avg_by_probe


def cluster_based_test(frate, compare='probe', cluster_entry_pval=0.05,
                       paired=False):
    '''Perform cluster-based tests on firing rate data.

    Performs cluster-based ANOVA on firing rate to test, for example,
    category-selectivity of the neurons. Currently t

    Parameters
    ----------
    frate : xarray.DataArray
        Xarray with spike rate  or spike density containing
        observations as the first dimension (for example trials for
        between-trials analysis or cells for between-cells analysis).
        If you have both cells and trials then the cell should already be
        selected, via ``frate.isel(cell=0)`` for example or the trials
        dimension should be averaged (for example ``frate.mean(dim='trial')``).
    compare : str
        Dimension labels specified for ``'trial'`` dimension that constitutes
        categories to test selectivity for.
    cluster_entry_pval : float
        p value used as a cluster-entry threshold. The default is ``0.05``.

    Returns
    -------
    stats : numpy.ndarray
        Anova F statistics for every timepoint.
    clusters : list of numpy.ndarray
        List of cluster memberships.
    pval : numpy.ndarray
        List of p values from anova.
    '''
    import warnings
    from scipy.stats.distributions import f
    from mne.stats import permutation_cluster_test

    if paired:
        from mne.stats import f_mway_rm

        def stat_fun(*args):
            data = np.stack(args, axis=1)
            n_factors = data.shape[1]
            fval, _ = f_mway_rm(data, factor_levels=[n_factors],
                                return_pvals=False)
            return fval
    else:
        from scipy.stats import f_oneway
        def stat_fun(*args):
            fval, _ = f_oneway(*args)
            return fval

    # calculate F anova threshold
    obs_dim = frate.dims[0]
    if paired:
        n_categories = len(np.unique(frate.coords[compare]))
        n_trials = len(frate.coords[obs_dim])
    else:
        categories, counts = np.unique(
            frate.coords[compare], return_counts=True)
        n_categories, n_trials = len(categories), counts.sum()

    p_thresh = cluster_entry_pval
    dfn = n_categories - 1
    dfd = n_trials - n_categories
    threshold = f.ppf(1. - p_thresh, dfn, dfd)

    # split data into probe groups
    arrays = [arr.values for _, arr in frate.groupby(compare)]

    # compute ANOVA cluster-based analysis
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stat, clusters, pval, _ = permutation_cluster_test(
            arrays, threshold=threshold, n_permutations=1000,
            stat_fun=stat_fun, out_type='mask', verbose=False)

    return stat, clusters, pval


# TODO: add offsets to spike-centered neurons
def spike_centered_windows(spk, cell_idx, arr, time, sfreq, winlen=0.1):
    from borsar.utils import find_index

    spike_centered = list()
    _, half_win = _symmetric_window_samples(winlen, sfreq)
    winlims = np.array([-half_win, half_win + 1])[None, :]
    lims = [0, len(time)]
    tri_is_ok = np.zeros(len(spk.trial[cell_idx]), dtype='bool')

    n_tri = max(spk.trial[cell_idx])
    for tri_idx in range(n_tri):
        sel = spk.trial[cell_idx] == tri_idx
        if sel.any():
            tms = spk.time[cell_idx][sel]
            if len(tms) < 1:
                continue

            closest_smp = find_index(time, tms)
            twins = closest_smp[:, None] + winlims
            good = ((twins >= lims[0]) & (twins <= lims[1])).all(axis=1)
            twins = twins[good]
            tri_is_ok[sel] = good

            for twin in twins:
                sig_part = arr[:, tri_idx, twin[0]:twin[1]]
                spike_centered.append(sig_part)

    spike_centered = np.stack(spike_centered, axis=1)
    tri = spk.trial[cell_idx][tri_is_ok]
    return spike_centered, tri


def spike_xcorr_density(spk, cell_idx, picks=None, sfreq=500, winlen=0.1,
                        kernel_winlen=0.025):

    # create kernel
    gauss_sd = kernel_winlen / 6 * sfreq
    win_smp, trim = _symmetric_window_samples(kernel_winlen, sfreq)
    kernel = _gauss_kernel_samples(win_smp, gauss_sd) * sfreq

    # calculate spike density
    picks = _deal_with_picks(spk, picks)
    tms, cnt = _spike_density(spk, picks=picks, sfreq=sfreq, kernel=kernel)
    cnt = cnt.transpose((1, 0, 2))

    # cut out spike-centered windows
    windows, tri = spike_centered_windows(
        spk, cell_idx, cnt, tms, sfreq, winlen=winlen)

    # correct autocorrelation if present:
    if cell_idx in picks:
        idx = np.where(np.asarray(picks) == cell_idx)[0][0]
        trim = int((windows.shape[-1] - len(kernel)) / 2)
        windows[idx, :, trim:-trim] -= kernel

    # turn to xarray
    t_per_smp = 1 / sfreq
    win_diff = [-winlen / 2, winlen / 2]
    time = np.arange(win_diff[0], win_diff[1] + 0.01 * t_per_smp,
                     step=t_per_smp)
    cell_names = [spk.cell_names[idx] for idx in picks]
    xcorr = _turn_spike_rate_to_xarray(time, windows, spk, tri=tri,
                                       cell_names=cell_names)

    return xcorr


# TODO: add shift predictor
def spike_xcorr_elephant(spk, cell_idx1, cell_idx2, sfreq=500, winlen=0.1,
                         kernel_winlen=0.025, shift_predictor=False):
    from scipy.signal import correlate
    import quantities as pq
    from elephant.conversion import BinnedSpikeTrain
    from elephant.spike_train_correlation import cross_correlation_histogram

    # create kernel
    gauss_sd = kernel_winlen / 6 * sfreq
    win_smp, trim = _symmetric_window_samples(kernel_winlen, sfreq)
    kernel = _gauss_kernel_samples(win_smp, gauss_sd) * sfreq

    # bin spikes
    binsize = 1 / sfreq
    spk1 = spk.to_neo(0)
    spk2 = spk.to_neo(1)
    bst1 = BinnedSpikeTrain(spk1, bin_size=binsize * pq.s)
    bst2 = BinnedSpikeTrain(spk2, bin_size=binsize * pq.s)

    cch_list = list()
    n_tri = bst1.shape[0]
    if shift_predictor:
        n_tri -= 1

    for tri in range(n_tri):
        if not shift_predictor:
            tri1, tri2 = tri, tri
        else:
            tri1, tri2 = tri, tri + 1

        cch, lags = cross_correlation_histogram(
            bst1[tri1], bst2[tri2], window=[-50, 50], kernel=kernel)
        cch_list.append(np.array(cch)[:, 0])

    # add last trial if shift predictor
    if shift_predictor:
        cch, lags = cross_correlation_histogram(
            bst1[-1], bst2[-2], window=[-50, 50], kernel=kernel)
        cch_list.append(np.array(cch)[:, 0])

    cch_list = np.stack(cch_list, axis=0)
    lags = lags * binsize

    cell_name = '{}-{}'.format(spk.cell_names[cell_idx1],
                               spk.cell_names[cell_idx2])
    cch = _turn_spike_rate_to_xarray(lags, cch_list[None, :], spk,
                                     cell_names=[cell_name])
    return cch


class Spikes(object):
    def __init__(self, timestamps, sfreq, cell_names=None, metadata=None,
                 cellinfo=None):
        '''Create ``Spikes`` object for convenient storage, analysis and
        visualization of spikes data.

        Parameters
        ----------
        timestamps : listlike of np.ndarray
            List of arrays, where each array contains spike timestamps for one
            cell (neuron).
        sfreq : float
            Sampling frequency of the timestamps. For example in Neuralynx
            system one timestamp occurs once per microsecond, so the sampling
            frequency is one million (``1e6``).
        cell_names : list of str | None
            String identifiers of cells. First string corresponds to first
            cell, that is ``time[0]`` and ``trial[0]`` (and so forth).
            Optional, the default (``None``) names the first cell
            ``'cell000'``, the second cell ``'cell001'`` and so on.
        metadata : pandas.DataFrame | None
            DataFrame with trial-level metadata.
        cellinfo : pandas.DataFrame | None
            Additional cell information.
        '''
        n_cells = len(timestamps)
        self.timestamps = timestamps
        self.sfreq = sfreq

        if cell_names is None:
            cell_names = np.array(['cell{:03d}'.format(idx)
                                   for idx in range(n_cells)])

        self.cell_names = cell_names
        self.metadata = metadata
        self.cellinfo = cellinfo

    def __repr__(self):
        '''Text representation of SpikeEpochs.'''
        n_cells = len(self.cell_names)
        avg_spikes = np.mean([len(x) for x in self.timestamps])
        msg = '<Spikes, {} cells, {:.1f} spikes/cell on average>'
        return msg.format(n_cells, avg_spikes)

    def epoch(self, events, event_id=None, tmin=-0.2, tmax=1.):
        '''Epoch spikes with respect to selected events.

        Parameters
        ----------
        events : numpy.ndarray
            ``n_events x 2`` or ``n_events x 3`` array of event timestamps.
            The first column should contain event timestamp (sample) and the
            second - event type (integer).
        event_id : None | int | list=like of int
            Event types to use in epoching. The default, ``None``, uses all
            events from ``events`` array in epoching.
        tmin : float
            Epoch start in seconds with respect to event onset. Default to
            ``-0.2``.
        tmax : float
            Epoch start in seconds with respect to event onset. Default to
            ``-0.2``.

        Returns
        -------
        spk : SpikeEpochs
            Epoched spikes.
        '''
        # event_id support
        if event_id is not None:
            use_events = np.in1d(events[:, -1], event_id)
            events = events[use_events, :]

        n_neurons = len(self.timestamps)
        trial, time = list(), list()
        event_times = events[:, 0] / self.sfreq

        for neuron_idx in range(n_neurons):
            tri, tim = _epoch_spikes(self.timestamps[neuron_idx] / self.sfreq,
                                     event_times, tmin, tmax)
            trial.append(tri)
            time.append(tim)

        spk = SpikeEpochs(time, trial, time_limits=[tmin, tmax],
                          cell_names=self.cell_names, cellinfo=self.cellinfo)

        if self.metadata is not None:
            if spk.n_trials == self.metadata.shape[0]:
                spk.metadata = self.metadata
            else:
                pass
                # raise warning ...

        return spk
