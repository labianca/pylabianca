from typing import Type
import numpy as np
import pandas as pd

from .utils import _deal_with_picks, _turn_spike_rate_to_xarray
from .spike_rate import compute_spike_rate, _spike_density
from .spike_distance import compare_spike_times


# TODO:
# - [ ] index by trial?
# - [ ] object of type 'SpikeEpochs' has no len() !
class SpikeEpochs():
    def __init__(self, time, trial, time_limits=None, n_trials=None,
                 waveform=None, cell_names=None, metadata=None, cellinfo=None):
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
        waveform : list of numpy ndarrays | None
            List of spikes x samples waveform arrays.
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
            n_trials = max(max(tri) + 1 if len(tri) > 0 else 0
                           for tri in self.trial)
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

        if waveform is not None:
            _check_waveforms(self.time, waveform)

        self.n_trials = n_trials
        self.waveform = waveform
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

    # TODO: consider if it is better to return number of epochs?
    def __len__(self):
        '''Return the number of neurons in SpikeEpochs.'''
        return len(self.time)

    def pick_cells(self, picks=None, query=None):
        '''Select cells by name or index. Operates in-place.

        Parameters
        ----------
        picks : int | str | listlike of int | list of str | None
            Cell names or indices to select.
        query : str | None
            Query for ``.cellinfo`` - to pick cells by their properties, not
            names or indices. Used only when ``picks`` is ``None``.
        '''
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
        if self.waveform is not None:
            self.waveform = [self.waveform[ix] for ix in picks]

        return self

    def drop_cells(self, picks):
        '''Drop cells by index. Operates in-place.

        Parameters
        ----------
        picks : int | str | listlike of int
            Cell  indices to drop.
        '''
        all_idx = np.arange(len(self))
        is_dropped = np.in1d(all_idx, picks)
        retain_idx = np.where(~is_dropped)[0]
        return self.pick_cells(retain_idx)

    def crop(self, tmin=None, tmax=None):
        '''Confine time range to specified limit. Operates in-place.

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
        tmin : float | None
            Time start in seconds. Default to trial start if ``tmin`` is ``None``.
        tmax : float | None
            Time end in seconds. Default to trial end if ``tmax`` is ``None``.
        backend : str
            Execution backend. Can be ``'numpy'`` or ``'numba'``.

        Returns
        -------
        frate : xarray.DataArray
            Xarray with following labeled dimensions: cell, trial, time.
        '''
        return compute_spike_rate(self, picks=picks, winlen=winlen, step=step,
                                  tmin=tmin, tmax=tmax, backend=backend)

    def spike_density(self, picks=None, winlen=0.3, gauss_sd=None, fwhm=None,
                      sfreq=500.):
        '''Compute spike density by convolving spikes with a gaussian kernel.

        Parameters
        ----------
        picks : int | listlike of int | None
            The neuron indices to use in the calculations. The default
            (``None``) uses all cells.
        winlen : float
            Length of the gaussian kernel window in seconds. Defaults to 0.3.
            Standard deviation of the gaussian kernel (``gauss_sd``), if
            unspecified (``gauss_sd=None``, default), is set as one sixth of
            the window length.
        gauss_sd : float | None
            Standard deviation of the gaussian kernel. By default it is set to
            ``winlen / 6``.
        fwhm : float | None
            Full width at half maximum of the gaussian kernel in seconds. If
            not ``None`` it overrides ``winlen`` and ``gauss_sd``.
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
                                  gauss_sd=gauss_sd, fwhm=fwhm, sfreq=sfreq)
        xarr = _turn_spike_rate_to_xarray(
            tms, cnt.transpose((1, 0, 2)), self,
            cell_names=self.cell_names[picks])
        return xarr

    def n_spikes(self, per_epoch=False):
        """Calculate number of spikes per cell (per epoch).

        Parameters
        ----------
        per_epoch: bool
            Whether to calculate number of spikes per cell splitting between
            epochs. If ``True`` the output is a ``cell x epochs`` xarray with
            additional trial info from ``.metadata`` attribute. Defaults to
            ``False``.

        Returns
        -------
        n_spikes : numpy.array | xarray.DataArray
            Number of spikes per cell. If ``per_epoch=False`` the output is
            a numpy array of spike numbers for consecutive cells. If
            ``per_epoch`` is ``True`` the output is a ``cell x epochs`` xarray
            with additional trial info from ``.metadata`` attribute.
        """
        return _n_spikes(self, per_epoch=per_epoch)

    # TODO:
    # - [ ] use `group` from sarna in looping through trials
    #       for faster execution...
    def to_neo(self, cell_idx, join=False, sep_time=0.):
        '''Turn spikes of given cell into neo.SpikeTrain format.

        Parameters
        ----------
        cell_idx : int
            Index of the cell to turn into neo.SpikeTrain format.
        join : bool | str
            Whether and how to join all the trials into a single
            neo.SpikeTrain. Defaults to ``False`` which does not perform
            any joining. If ``join=True`` or ``join='concat'``, the trials
            are concatenated into a single neo.SpikeTrain using ``sep_time``
            as time separation between consecutive trials. If ``join='pool'``,
            all the spikes are pooled and sorted with respect to their time
            (irrespective of trial number).

        Returns
        -------
        spikes : neo.SpikeTrain | list of neo.SpikeTrain
            If ``join=True``, a single neo.SpikeTrain is returned with spikes
            from separate trials concatenated (or pooled and sorted if
            ``join='pool'``. Otherwise (``join=False``) a list of
            neo.SpikeTrain objects is returned.
        '''
        import neo
        from quantities import s

        if isinstance(join, bool) and join:
            join = 'concat'

        if not join:
            spikes = list()
            trials = range(self.n_trials)
            for tri in trials:
                times = self.time[cell_idx][self.trial[cell_idx] == tri]
                spiketrain = neo.SpikeTrain(
                    times * s, t_stop=self.time_limits[1],
                    t_start=self.time_limits[0])
                spikes.append(spiketrain)
        elif join == 'concat':
            trial_len = self.time_limits[1] - self.time_limits[0]
            full_sep = trial_len + sep_time
            new_times = self.time[cell_idx] + full_sep * self.trial[cell_idx]
            t_stop = trial_len * self.n_trials + sep_time * (self.n_trials - 1)

            spikes = neo.SpikeTrain(
                new_times * s, t_stop=t_stop * s, t_start=self.time_limits[0])
        elif join == 'pool':
            times = np.sort(self.time[cell_idx])
            spikes = neo.SpikeTrain(
                times * s, t_stop=self.time_limits[1],
                t_start=self.time_limits[0])
        return spikes

    # TODO: return xarray?
    def to_raw(self, picks=None, sfreq=500.):
        '''Turn epoched spike timestamps into binned continuous representation.

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
            ``trials x cells x timesamples`` array with binary spike
            information.
        '''
        return _spikes_to_raw(self, picks=picks, sfreq=sfreq)

    def __getitem__(self, selection):
        '''Select trials using an array of integers or metadata query.'''
        if isinstance(selection, str):
            if self.metadata is None:
                raise TypeError('metadata cannot be ``None`` when selecting '
                                'trials with a query.')
            # treat as pandas-style query
            new_metadata = self.metadata.query(selection)
            tri_idx = new_metadata.index.values
        elif isinstance(selection, (np.ndarray, list, tuple)):
            selection = np.asarray(selection)
            assert np.issubdtype(selection.dtype, np.integer)

            if self.metadata is not None:
                new_metadata = self.metadata.iloc[selection, :]
            tri_idx = selection
        else:
            raise TypeError('Currently only string queries are allowed to '
                            'select elements of SpikeEpochs')

        newtime, newtrial = list(), list()
        new_metadata = new_metadata.reset_index(drop=True)

        has_waveform = self.waveform is not None
        waveform = list() if has_waveform else None

        # for each cell select relevant trials:
        for cell_idx in range(len(self.trial)):
            cell_tri = self.trial[cell_idx]
            sel = np.in1d(cell_tri, tri_idx)
            newtime.append(self.time[cell_idx][sel])

            this_tri = (cell_tri[sel, None] == tri_idx[None, :]).argmax(axis=1)
            newtrial.append(this_tri)

            if has_waveform:
                waveform.append(self.waveform[cell_idx][sel])

        new_cellinfo = None if self.cellinfo is None else self.cellinfo.copy()
        return SpikeEpochs(newtime, newtrial, time_limits=self.time_limits,
                           n_trials=new_metadata.shape[0],
                           cell_names=self.cell_names.copy(),
                           metadata=new_metadata, cellinfo=new_cellinfo,
                           waveform=waveform)

    def plot_waveform(self, pick=0, upsample=False, ax=None, labels=True):
        '''Plot waveform heatmap for one cell.

        Parameters
        ----------
        pick : int
            Cell index to plot waveform for.
        upsample : bool | float
            Whether to upsample the waveform (defaults to ``False``). If
            ``True`` the waveform is upsampled by a factor of three. Can also
            be a value to specify the upsampling factor.
        ax : matplotlib.Axes | None
            Axis to plot to. By default opens a new figure.
        '''
        from .viz import plot_waveform
        return plot_waveform(self, pick=pick, upsample=upsample, ax=ax,
                             labels=labels)


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
    idx : numpy array
        Indices of spikes that were retained. Depending on the epoching, some
        spikes may be duplicated.
    '''
    trial = list()
    time = list()
    idx = list()

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

            idx.append(np.where(msk)[0] + first_idx)
        t_idx = first_idx

    if len(trial) > 0:
        trial = np.concatenate(trial)
        time = np.concatenate(time)
        idx = np.concatenate(idx)
    else:
        trial = np.array([])
        time = np.array([])

    return trial, time, idx


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


class Spikes(object):
    def __init__(self, timestamps, sfreq, cell_names=None, metadata=None,
                 cellinfo=None, waveform=None):
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
        waveform : list of np.ndarray
            List of spikes x samples waveform arrays.
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

        if waveform is not None:
            _check_waveforms(timestamps, waveform)
            self.waveform = waveform
        else:
            self.waveform = None

    def __repr__(self):
        '''Text representation of SpikeEpochs.'''
        n_cells = len(self.cell_names)
        avg_spikes = np.mean([len(x) for x in self.timestamps])
        msg = '<Spikes, {} cells, {:.1f} spikes/cell on average>'
        return msg.format(n_cells, avg_spikes)

    # TODO: return idx from _epoch_spikes only when self.waveform is not None
    # TODO: time and consider speeding up
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

        has_waveform = self.waveform is not None
        waveforms = list() if has_waveform else None

        for neuron_idx in range(n_neurons):
            tri, tim, idx = _epoch_spikes(
                self.timestamps[neuron_idx] / self.sfreq, event_times,
                tmin, tmax)
            trial.append(tri)
            time.append(tim)

            if has_waveform:
                waveforms.append(self.waveform[neuron_idx][idx, :])

        spk = SpikeEpochs(time, trial, time_limits=[tmin, tmax],
                          cell_names=self.cell_names, cellinfo=self.cellinfo,
                          n_trials=len(events), waveform=waveforms)

        # TODO: this should be removed later on, as Spike metadata should not
        #       be supported, metadata should be provided during or after
        #       epoching
        if self.metadata is not None:
            if spk.n_trials == self.metadata.shape[0]:
                spk.metadata = self.metadata
            else:
                pass
                # raise warning ...

        return spk

    # TODO: ability to get a shallow copy might also be useful
    # TODO: refactor with `.copy()` in SpikeEpochs
    def copy(self):
        '''Return a deep copy of the object.'''
        from copy import deepcopy
        return deepcopy(self)

    def __len__(self):
        '''Return the number of neurons in SpikeEpochs.'''
        return len(self.timestamps)

    # TODO: refactor out common parts with SpikeEpochs.pick_cells
    def pick_cells(self, picks=None, query=None):
        '''Select cells by name or index. Operates in-place.

        Parameters
        ----------
        picks : int | str | listlike of int | list of str | None
            Cell names or indices to select.
        query : str | None
            Query for ``.cellinfo`` - to pick cells by their properties, not
            names or indices. Used only when ``picks`` is ``None``.

        Returns
        -------
        spk : Spikes
            Selected units in a Spikes object.
        '''
        if picks is None and query is None:
            return self

        if picks is None and query is not None:
            assert self.cellinfo is not None
            cellinfo_sel = self.cellinfo.query(query)
            picks = cellinfo_sel.index.values
        else:
            picks = _deal_with_picks(self, picks)

        self.timestamps = [self.timestamps[ix] for ix in picks]
        self.cell_names = self.cell_names[picks].copy()
        if self.cellinfo is not None:
            self.cellinfo = self.cellinfo.loc[picks, :].reset_index(drop=True)
        if self.waveform is not None:
            self.waveform = [self.waveform[ix] for ix in picks]

        return self

    # TODO: DRY with SpikeEpochs
    def drop_cells(self, picks):
        '''Drop cells by index. Operates in-place.

        Parameters
        ----------
        picks : int | str | listlike of int
            Cell  indices to drop.
        '''
        all_idx = np.arange(len(self))
        is_dropped = np.in1d(all_idx, picks)
        retain_idx = np.where(~is_dropped)[0]
        return self.pick_cells(retain_idx)

    def n_spikes(self):
        """Calculate number of spikes per cell.

        Returns
        -------
        n_spikes : numpy.array
            Number of spikes per cell.
        """
        return _n_spikes(self)

    def sort(self, by=None):
        '''Sort cells. Operates in-place.

        The units are by default sorted by channel and cluster id information
        contained in dataframe stored in ``.cellinfo`` attribute.

        Parameters
        ----------
        by : str | list of str | None
            If ``None`` (default) the units are sorted by channel and cluster
            information contained in ``.cellinfo``.
            If string or list of strings - name/names of ``.cellinfo`` columns
            to sort by.
            Defaults to ``None``.

        Returns
        -------
        spk : Spikes
            Sorted Spikes.
        '''
        self = _sort_spikes(self, by)
        return self

    def plot_waveform(self, pick=0, upsample=False, ax=None, labels=True):
        '''Plot waveform heatmap for one cell.

        Parameters
        ----------
        pick : int
            Cell index to plot waveform for.
        upsample : bool | float
            Whether to upsample the waveform (defaults to ``False``). If
            ``True`` the waveform is upsampled by a factor of three. Can also
            be a value to specify the upsampling factor.
        ax : matplotlib.Axes | None
            Axis to plot to. By default opens a new figure.

        Returns
        -------
        ax : matplotlib.Axes
            Axis with waveform heatmap.
        '''
        from .viz import plot_waveform
        return plot_waveform(self, pick=pick, upsample=upsample, ax=ax,
                             labels=labels)

    def to_epochs(self, pad_timestamps=10_000):
        '''Turn Spike object into one-epoch SpikeEpochs representation.'''
        min_stamp = (int(min([min(x) for x in self.timestamps]))
                     - pad_timestamps)
        max_stamp = (int(max([max(x) for x in self.timestamps]))
                     + pad_timestamps)
        stamp_diff = max_stamp - min_stamp
        s_len = stamp_diff / self.sfreq

        events_fake = np.array([[min_stamp, 0, 123]])
        tmin, tmax = 0, s_len
        spk_epochs = self.epoch(events_fake, event_id=123,
                                tmin=tmin, tmax=tmax)
        return spk_epochs

    def merge(self, picks):
        '''Merge spikes from multiple cells into one. Operates in-place.

        Parameters
        ----------
        picks : list of int
            Indices of cells to merge.

        Returns
        -------
        spk : Spikes
            Modified Spikes object. The return values is to allow for chaining,
            as ``spk.merge(picks)`` operates in-place.
        '''

        picks = _deal_with_picks(self, picks)
        picks = list(picks)
        picks.sort()
        picks = picks[::-1]

        # pop timestamps in reverse order (so that indices are valid) and add
        # pop also ch_names and waveform if present
        for idx in picks[:-1]:
            if isinstance(self.cell_names, list):
                self.cell_names.pop(idx)
            else:
                self.cell_names = np.delete(self.cell_names, idx, axis=0)

            stamps = self.timestamps.pop(idx)
            self.timestamps[picks[-1]] = np.concatenate(
                (self.timestamps[picks[-1]], stamps))

            if self.waveform is not None:
                waveform = self.waveform.pop(idx)
                self.waveform[picks[-1]] = np.concatenate(
                    (self.waveform[picks[-1]], waveform))

        # sort spikes and waveforms
        ordering = np.argsort(self.timestamps[picks[-1]])
        self.timestamps[picks[-1]] = self.timestamps[picks[-1]][ordering]
        if self.waveform is not None:
            self.waveform[picks[-1]] = self.waveform[picks[-1]][ordering]

        if self.cellinfo is not None:
            # drop all but the lowest index from cellinfo
            # (agg info in cluster field?)
            self.cellinfo = self.cellinfo.drop(
                index=picks[:-1]).reset_index(drop=True)

        return self

    def to_matlab(self, path, format='osort_mm'):
        '''Save Spikes object to a matlab file in the desired format.'''
        assert format in ['fieldtrip', 'osort_mm'], 'Unknown format.'
        if format == 'fieldtrip':
            raise NotImplementedError('Sorry this is not implemented yet.')
        elif format == 'osort_mm':
            from .io import _save_spk_to_mm_matlab_format as write_spikes
        write_spikes(self, path)


def _check_waveforms(times, waveform):
    '''Safety checks for waveform data.'''
    assert len(times) == len(waveform)
    n_spikes_times = np.array([len(x) for x in times])
    n_spikes_waveform = np.array([x.shape[0] for x in waveform])
    assert (n_spikes_times == n_spikes_waveform).all()


def concatenate_spikes(spk_list, sort=True, relabel_cell_names=True):
    '''Concatenate list of spike objects into one.

    Parameters
    ----------
    spk_list : list of Spikes
        List of Spikes objects to concatenate.
    sort : bool | str | list of str
        If boolean: whether to sort the concatenated units. The units are then
        sorted by channel and cluster contained in ``.cellinfo``.
        If string or list of strings - name/names of ``.cellinfo`` columns to
        sort by. Defaults to ``True``.
    relabel_cell_names : bool
        Whether to relabel cell names to correspond to cell index.
        Defaults to ``True``.

    Returns
    -------
    spk : Spikes
        Concatenated spikes object.
    '''
    assert len(spk_list) > 0
    for spk in spk_list:
        assert isinstance(spk, Spikes), ('Not all elements in spk_list are '
                                         'Spikes objects.')

    if len(spk_list) == 1:
        return spk_list[0]

    spk = spk_list[0].copy()
    has_cellinfo = spk.cellinfo is not None
    has_waveform = spk.waveform is not None

    if has_cellinfo:
        cell_infos = [spk.cellinfo.copy()]

    for spk_add in spk_list[1:]:
        # cell names
        spk.cell_names = np.concatenate(
            [spk.cell_names, spk_add.cell_names])

        # timestamps
        spk.timestamps.extend(spk_add.timestamps)

        # cellinfo
        if has_cellinfo:
            cell_infos.append(spk_add.cellinfo)

        # waveform
        if has_waveform:
            spk.waveform.extend(spk_add.waveform)

        # metadata - only for SpikeEpochs
        # this attrib should not be present in Spikes

    if has_cellinfo:
        spk.cellinfo = pd.concat(cell_infos).reset_index(drop=True)

    if sort:
        spk = spk.sort() if isinstance(sort, bool) else spk.sort(by=sort)

    if relabel_cell_names:
        n_cells = len(spk)
        spk.cell_names = np.array(['cell{:03d}'.format(idx)
                                   for idx in range(n_cells)])

    return spk


def _sort_spikes(spk, by=None, inplace=True):
    '''Sort units by channel and cluster id or other columns in cellinfo.'''
    by = ['channel', 'cluster'] if by is None else by

    # the tests below were written by GitHub copilot entirely!
    if isinstance(by, str):
        by = [by]
    assert isinstance(by, list)
    assert all([isinstance(x, str) for x in by])
    assert all([x in spk.cellinfo.columns for x in by])

    if not inplace:
        spk = spk.copy()

    cellinfo_sorted = spk.cellinfo.sort_values(
        by=by, axis='index')
    cells_order = cellinfo_sorted.index.to_numpy()
    spk.pick_cells(cells_order)

    return spk


def _n_spikes(spk, per_epoch=False):
    """Calculate number of spikes."""
    if not per_epoch:
        if isinstance(spk, Spikes):
            return np.array([len(x) for x in spk.timestamps])
        elif isinstance(spk, SpikeEpochs):
            return np.array([len(x) for x in spk.time])
        else:
            raise TypeError("`spk` has to be an instance of Spikes or"
                            f" SpikeEpochs, got {type(spk)}.")

    else:
        if not isinstance(spk, SpikeEpochs):
            raise TypeError("When `per_epoch=True`, `spk` has to be an "
                            f"instance of SpikeEpochs, got {type(spk)}.")
        tmin, tmax = spk.time_limits
        winlen = tmax - tmin

        # FIX: this normalizes per second, we don't want that in n_spikes
        frate = compute_spike_rate(spk, step=False, tmin=tmin, tmax=tmax)
        n_spk = (frate.values * winlen).astype('int')
        return n_spk
