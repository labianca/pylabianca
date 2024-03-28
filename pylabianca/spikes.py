from typing import Type
import numpy as np
import pandas as pd

from .utils import (_deal_with_picks, _turn_spike_rate_to_xarray,
                    _get_trial_boundaries, _validate_spike_epochs_input,
                    _validate_cellinfo)
from .spike_rate import compute_spike_rate, _spike_density, _add_frate_info
from .spike_distance import compare_spike_times, xcorr_hist


class SpikeEpochs():
    def __init__(self, time, trial, time_limits=None, n_trials=None,
                 waveform=None, waveform_time=None, cell_names=None,
                 metadata=None, cellinfo=None, timestamps=None):
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
        waveform_time : np.ndarray | None
            One-dimensional array of time values in milliseconds for
            consecutive samples of the waveform.
        cell_names : list of str | None
            String identifiers of cells. First string corresponds to first
            cell, that is ``time[0]`` and ``trial[0]`` (and so forth).
            Optional, the default (``None``) names the first cell
            ``'cell000'``, the second cell ``'cell001'`` and so on.
        metadata : pandas.DataFrame
            DataFrame with trial-level metadata.
        cellinfo : pandas.DataFrame
            DataFrame with additional information about the cells.
        timestamps : listlike of np.ndarray | None
            Original spike timestamps. Should be in the same format as ``time``
            and ``trial`` arguments.
        '''
        _validate_spike_epochs_input(time, trial)

        self.time = time
        self.trial = trial

        if time_limits is None:
            tmin = min([min(x) for x in time])
            tmax = max([max(x) for x in time]) + 1e-6
            time_limits = np.array([tmin, tmax])
        self.time_limits = time_limits

        if n_trials is None:
            n_trials = int(max(max(tri) + 1 if len(tri) > 0 else 0
                           for tri in self.trial))
        if cell_names is None:
            n_cells = len(time)
            cell_names = np.array(['cell{:03d}'.format(idx)
                                   for idx in range(n_cells)])
        else:
            cell_names = np.asarray(cell_names)
            assert len(cell_names) == len(time)

        if metadata is not None:
            assert isinstance(metadata, pd.DataFrame)
            assert metadata.shape[0] == n_trials

        n_cells = len(self.time)
        self._cellinfo = _validate_cellinfo(self, cellinfo)

        if waveform is not None:
            _check_waveforms(self.time, waveform, waveform_time)

        self.n_trials = n_trials
        self.waveform = waveform
        self.waveform_time = waveform_time
        self.cell_names = cell_names
        self.cellinfo = cellinfo
        self.timestamps = timestamps
        self.metadata = metadata

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

    def __len__(self):
        '''Return the number of epochs in SpikeEpochs.'''
        return self.n_trials

    def n_units(self):
        '''Return the number of units in SpikeEpochs.'''
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
        return _pick_cells(self, picks=picks, query=query)

    def drop_cells(self, picks):
        '''Drop cells by index. Operates in-place.

        Parameters
        ----------
        picks : int | str | listlike of int
            Cell  indices to drop.
        '''
        return _drop_cells(self, picks)

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

        has_waveform = self.waveform is not None
        has_timestamps = self.timestamps is not None

        for cell_idx in range(len(self.time)):
            sel = (self.time[cell_idx] >= tmin) & (self.time[cell_idx] <= tmax)
            self.time[cell_idx] = self.time[cell_idx][sel]
            self.trial[cell_idx] = self.trial[cell_idx][sel]
            if has_waveform:
                self.waveform[cell_idx] = self.waveform[cell_idx][sel, :]
            if has_timestamps:
                self.timestamps[cell_idx] = self.timestamps[cell_idx][sel]
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
        xarr = _add_frate_info(xarr, dep='density')

        return xarr

    def xcorr(self, picks=None, picks2=None, sfreq=500., max_lag=0.2,
              bins=None, gauss_fwhm=None):
        """
        Calculate cross-correlation histogram.

        Parameters
        ----------
        picks : int | str | list-like of int | list-like of str | None
            List of cell indices or names to perform cross- and auto- correlations
            for. If ``picks2`` is ``None`` then all combinations of cells from
            ``picks`` will be used.
        picks2 : int | str | list-like of int | list-like of str | None
            List of cell indices or names to perform cross-correlations with.
            ``picks2`` is used as pairs for ``picks``. The interaction between
            ``picks`` and ``picks2`` is the following:
            * if ``picks2`` is ``None`` only ``picks`` is consider to contruct all
                combinations of pairs.
            * if ``picks2`` is not ``None`` and ``len(picks) == len(picks2)`` then
                pairs are constructed from successive elements of ``picks`` and
                ``picks2``. For example, if ``picks = [0, 1, 2]`` and
                ``picks2 = [3, 4, 5]`` then pairs will be constructed as
                ``[(0, 3), (1, 4), (2, 5)]``.
            * if ``picks2`` is not ``None`` and ``len(picks) != len(picks2)`` then
                all combinations of ``picks`` and ``picks2`` are used.
        sfreq : float
            Sampling frequency of the bins. The bin width will be ``1 / sfreq``
            seconds. Used only when ``bins`` is ``None``. Defaults to ``500.``.
        max_lag : float
            Maximum lag in seconds. Used only when ``bins is None``. Defaults
            to ``0.2``.
        bins : numpy array | None
            Array representing edges of the histogram bins. If ``None`` (default)
            the bins are constructed based on ``sfreq`` and ``max_lag``.
        gauss_fwhm : float | None
            Full-width at half maximum of the gaussian kernel to convolve the
            cross-correlation histograms with. Defaults to ``None`` which ommits
            convolution.

        Returns
        -------
        xcorr : xarray.DataArray
            Xarray DataArray of cross-correlation histograms. The first
            dimension is the cell pair, and the last dimension is correlation
            the lag. If the input is SpikeEpochs then the second dimension is
            the trial.
        """
        return xcorr_hist(
            self, picks=picks, picks2=picks2, sfreq=sfreq, max_lag=max_lag,
              bins=bins, gauss_fwhm=gauss_fwhm, backend='numpy'
        )

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

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, df):
        if df is not None:
            assert isinstance(df, pd.DataFrame)
            assert df.shape[0] == len(self)
            df = df.reset_index(drop=True)
        self._metadata = df

    @property
    def cellinfo(self):
        return self._cellinfo

    @cellinfo.setter
    def cellinfo(self, df):
        self._cellinfo = _validate_cellinfo(self, df)

    # TODO:
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

    # TODO: return xarray? / return mne.Epochs?
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

    def to_spiketools(self, picks=None):
        '''Convert pylabianca SpikeEpochs to list of arrays.

        Parameters
        ----------
        picks : None | list of ints
            Which units to convert. If ``None``, all units are converted.

        Returns
        -------
        inst : list of arrays
            List of arrays containing spike times. Each array corresponds to
            one trial. When multiple picks are provided, the output is a list
            of lists of arrays (where outermost list elements correspond to
            units).
        '''
        from .io import to_spiketools
        return to_spiketools(self, picks)

    def __getitem__(self, selection):
        '''Select trials using an array of int / bool or metadata query.'''
        if isinstance(selection, str):
            if self.metadata is None:
                raise TypeError('metadata cannot be ``None`` when selecting '
                                'trials with a query.')
            # treat as pandas-style query
            new_metadata = self.metadata.query(selection)
            tri_idx = new_metadata.index.values
        elif isinstance(selection, (np.ndarray, list, tuple)):
            selection = np.asarray(selection)
            int_sel = np.issubdtype(selection.dtype, np.integer)
            bool_sel = np.issubdtype(selection.dtype, np.bool_)
            assert int_sel or bool_sel

            if int_sel:
                if self.metadata is not None:
                    new_metadata = self.metadata.iloc[selection, :]
                tri_idx = selection
            else:
                if self.metadata is not None:
                    new_metadata = self.metadata.loc[selection, :]
                tri_idx = np.where(selection)[0]
        else:
            raise TypeError('Currently only string queries are allowed to '
                            'select elements of SpikeEpochs')

        new_time, new_trial = list(), list()
        if self.metadata is not None:
            new_metadata = new_metadata.reset_index(drop=True)
        else:
            new_metadata = None

        has_waveform = self.waveform is not None
        has_timestamps = self.timestamps is not None
        waveform = list() if has_waveform else None
        timestamps = list() if has_timestamps else None

        # for each cell select relevant trials:
        for cell_idx in range(len(self.trial)):
            cell_tri = self.trial[cell_idx]
            sel = np.in1d(cell_tri, tri_idx)
            new_time.append(self.time[cell_idx][sel])

            this_tri = (cell_tri[sel, None] == tri_idx[None, :]).argmax(axis=1)
            new_trial.append(this_tri)

            if has_waveform:
                waveform.append(self.waveform[cell_idx][sel])
            if has_timestamps:
                timestamps.append(self.timestamps[cell_idx][sel])

        new_cellinfo = None if self.cellinfo is None else self.cellinfo.copy()
        return SpikeEpochs(new_time, new_trial, time_limits=self.time_limits,
                           n_trials=len(tri_idx),
                           cell_names=self.cell_names.copy(),
                           metadata=new_metadata, cellinfo=new_cellinfo,
                           waveform=waveform, timestamps=timestamps)

    def plot_waveform(self, picks=None, upsample=False, ax=None, labels=True):
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
        return plot_waveform(self, picks=picks, upsample=upsample, ax=ax,
                             labels=labels, times=self.waveform_time)

    def apply(self, func, picks=None, args=None, kwargs=None):
        '''Apply a function to each cell and trial.

        Parameters
        ----------
        func : callable
            Function to apply to each cell.
        picks : int | str | list-like of int | list-like of str | None
            Cell indices or names to apply the function to. Optional, the
            default (``None``) applies the function to all cells.
        args : list | None
            Positional arguments to pass to the function. Defaults to ``None``.
        kwargs : dict | None
            Keyword arguments to pass to the function. Defaults to ``None``.

        Returns
        -------
        out : list
            List of outputs from the function.
        '''
        from .io import to_spiketools

        if args is None:
            args = list()
        if kwargs is None:
            kwargs = dict()

        picks = _deal_with_picks(self, picks)

        out = list()
        max_trials = self.n_trials
        for pick in picks:
            # ENH: add `to_spiketools()` method !
            # trials = self.to_spiketools(pick)
            trials = to_spiketools(self, pick)

            trial_list = list()
            for tri in trials:
                value = func(tri, *args, **kwargs)
                trial_list.append(value)
            # else:
            #     trial_list.append(np.nan)
            out.append(np.array(trial_list))

        out = np.stack(out, axis=0)

        # attempt to convert to xarray
        names = [self.cell_names[pick] for pick in picks]
        out = _turn_spike_rate_to_xarray(
            None, out, self, cell_names=names)

        return out


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
        above_lower = timestamps[t_idx:] > (event_times[epo_idx] + tmin)
        # first_idx = above_lower.argmax() + t_idx
        first_idx = np.where(above_lower)[0]
        if len(first_idx) > 0:
            first_idx = first_idx[0] + t_idx
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
    half_sample = sample_time / 2
    tmin, tmax = spk.time_limits
    times = np.arange(tmin, tmax + 0.01 * sample_time, step=sample_time)
    time_bins = np.concatenate(
        [times - half_sample, [times[-1] + half_sample]])

    n_cells = len(picks)
    n_times = len(times)
    trials_raw = np.zeros((spk.n_trials, n_cells, n_times), dtype='int')

    for idx, cell_idx in enumerate(picks):
        trial_boundaries, tri_num = _get_trial_boundaries(spk, cell_idx)
        for ix, this_tri in enumerate(tri_num):

            spike_times = spk.time[cell_idx][
                trial_boundaries[ix]:trial_boundaries[ix + 1]]
            trials_raw[this_tri, idx, :] = np.histogram(
                spike_times, bins=time_bins
            )[0]

    return times, trials_raw


class Spikes(object):
    def __init__(self, timestamps, sfreq, cell_names=None,
                 cellinfo=None, waveform=None, waveform_time=None):
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
        cellinfo : pandas.DataFrame | None
            Additional cell information.
        waveform : list of np.ndarray
            List of spikes x samples waveform arrays.
        waveform_time : np.ndarray | None
            One-dimensional array of time values in milliseconds for
            consecutive samples of the waveform.
        '''
        n_cells = len(timestamps)
        self.timestamps = timestamps
        self.sfreq = sfreq

        if cell_names is None:
            cell_names = np.array(['cell{:03d}'.format(idx)
                                   for idx in range(n_cells)])
        else:
            assert len(cell_names) == len(timestamps)

        self.cell_names = cell_names
        self._cellinfo = _validate_cellinfo(self, cellinfo)

        if waveform is not None:
            _check_waveforms(timestamps, waveform, waveform_time)
            self.waveform = waveform
            self.waveform_time = waveform_time
        else:
            self.waveform = None
            self.waveform_time = None

    def __repr__(self):
        '''Text representation of SpikeEpochs.'''
        n_cells = self.n_units()
        avg_spikes = np.mean([len(x) for x in self.timestamps])
        msg = '<Spikes, {} cells, {:.1f} spikes/cell on average>'
        return msg.format(n_cells, avg_spikes)

    def n_units(self):
        '''Return the number of units in Spikes.'''
        return len(self.timestamps)

    # TODO: return idx from _epoch_spikes only when self.waveform is not None?
    # TODO: time and consider speeding up
    def epoch(self, events, event_id=None, tmin=-0.2, tmax=1.,
              keep_timestamps=False):
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
        keep_timestamps : bool
            Whether to keep the original spike timestamps and store them in the
            epochs. Defaults to ``False``.

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
        timestamps = list() if keep_timestamps else None

        for neuron_idx in range(n_neurons):
            tri, tim, idx = _epoch_spikes(
                self.timestamps[neuron_idx] / self.sfreq, event_times,
                tmin, tmax)
            trial.append(tri)
            time.append(tim)

            if has_waveform:
                waveforms.append(self.waveform[neuron_idx][idx, :])

            if keep_timestamps:
                timestamps.append(self.timestamps[neuron_idx][idx])

        spk = SpikeEpochs(time, trial, time_limits=[tmin, tmax],
                          cell_names=self.cell_names, cellinfo=self.cellinfo,
                          n_trials=len(events), waveform=waveforms,
                          waveform_time=self.waveform_time,
                          timestamps=timestamps)

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
        return _pick_cells(self, picks=picks, query=query)

    def drop_cells(self, picks):
        '''Drop cells by index. Operates in-place.

        Parameters
        ----------
        picks : int | str | listlike of int
            Cell  indices to drop.
        '''
        return _drop_cells(self, picks)

    def n_spikes(self):
        """Calculate number of spikes per cell.

        Returns
        -------
        n_spikes : numpy.array
            Number of spikes per cell.
        """
        return _n_spikes(self)

    @property
    def cellinfo(self):
        return self._cellinfo

    @cellinfo.setter
    def cellinfo(self, df):
        self._cellinfo = _validate_cellinfo(self, df)

    def sort(self, by=None, inplace=True):
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
        inplace : bool
            Whether to sort the units in place. Defaults to ``True``.

        Returns
        -------
        spk : Spikes
            Sorted Spikes.
        '''
        self = _sort_spikes(self, by, inplace=inplace)
        return self

    def plot_waveform(self, picks=None, upsample=False, ax=None, labels=True):
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
        return plot_waveform(self, picks=picks, upsample=upsample, ax=ax,
                             labels=labels, times=self.waveform_time)

    def plot_isi(self, picks=None, unit='ms', bins=None, min_spikes=100,
                 max_isi=None, ax=None):
        '''Plot inter-spike intervals (ISIs).

        Parameters
        ----------
        spk : pylabianca.spikes.Spikes
            Spikes object to use.
        picks : int | str | list of int | list of str | None
            Which cells to plot. If ``None`` all cells are plotted.
        unit : str
            Time unit to use when plotting the ISIs. Can be ``'ms'`` or ``'s'``.
        bins : int | None
            Number of bins to use for the histograms. If ``None`` the number of
            bins is automatically determined.
        min_spikes : int
            Minimum number of spikes required to plot the ISI histogram.
        max_isi : float | None
            Maximum ISI time to plot. If ``None`` the maximum ISI is set to 0.1
            for ``unit == 's'`` and 100 for ``unit == 'ms'``.
        ax : matplotlib.Axes | None
            Axis to plot to. If ``None`` a new figure is created.

        Returns
        -------
        ax : matplotlib.Axes
            Axes with the plot.
        '''
        from .viz import plot_isi
        return plot_isi(self, picks=picks, unit=unit, bins=bins,
                        min_spikes=min_spikes, max_isi=max_isi, ax=ax)

    # TODO: pad_timestamps could be thrown away if we read / store recording
    #       start for example or allow another keyword argument for recording
    #       start timestamp.
    def to_epochs(self, pad_timestamps=0):
        '''Turn Spike object into one-epoch SpikeEpochs representation.

        Spike object does not know the start and end time of the recording, so
        when using ``.to_epochs()`` method it assumes that the earliest spike
        timestamp marks the start of the recording and has the time of 0.

        Parameters
        ----------
        pad_timestamps : int
            By default the earliest spike will have a time of `0.` seconds.
            ``pad_timestamps`` can be used to change that by setting the
            recording start as the earliest spike timestamp MINUS
            ``pad_timestamps``. Defaults to 0.
        '''
        min_stamp = (np.min([x[0] for x in self.timestamps])
                    - pad_timestamps)
        max_stamp = (np.max([x[-1] for x in self.timestamps])
                    + pad_timestamps)
        stamp_diff = max_stamp - min_stamp
        s_len = stamp_diff / self.sfreq

        tmin, tmax = 0, s_len

        time = [(x - min_stamp) / self.sfreq for x in self.timestamps]
        trial = [np.zeros(len(x), dtype=int) for x in self.timestamps]
        waveform = self.waveform.copy() if self.waveform is not None else None
        waveform_time = (self.waveform_time.copy()
                         if self.waveform_time is not None else None)
        cell_names = (None if self.cell_names is None
                      else self.cell_names.copy())
        cellinfo = None if self.cellinfo is None else self.cellinfo.copy()

        spk_epochs = SpikeEpochs(
            time, trial, time_limits=[tmin, tmax], n_trials=1,
            waveform=waveform, waveform_time=waveform_time,
            cell_names=cell_names, cellinfo=cellinfo)
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

    def xcorr(self, picks=None, picks2=None, sfreq=500., max_lag=0.2,
              bins=None, gauss_fwhm=None, backend='auto'):
        """
        Calculate cross-correlation histogram.

        Parameters
        ----------
        picks : int | str | list-like of int | list-like of str | None
            List of cell indices or names to perform cross- and auto- correlations
            for. If ``picks2`` is ``None`` then all combinations of cells from
            ``picks`` will be used.
        picks2 : int | str | list-like of int | list-like of str | None
            List of cell indices or names to perform cross-correlations with.
            ``picks2`` is used as pairs for ``picks``. The interaction between
            ``picks`` and ``picks2`` is the following:
            * if ``picks2`` is ``None`` only ``picks`` is consider to contruct all
                combinations of pairs.
            * if ``picks2`` is not ``None`` and ``len(picks) == len(picks2)`` then
                pairs are constructed from successive elements of ``picks`` and
                ``picks2``. For example, if ``picks = [0, 1, 2]`` and
                ``picks2 = [3, 4, 5]`` then pairs will be constructed as
                ``[(0, 3), (1, 4), (2, 5)]``.
            * if ``picks2`` is not ``None`` and ``len(picks) != len(picks2)`` then
                all combinations of ``picks`` and ``picks2`` are used.
        sfreq : float
            Sampling frequency of the bins. The bin width will be ``1 / sfreq``
            seconds. Used only when ``bins`` is ``None``. Defaults to ``500.``.
        max_lag : float
            Maximum lag in seconds. Used only when ``bins is None``. Defaults
            to ``0.2``.
        bins : numpy array | None
            Array representing edges of the histogram bins. If ``None`` (default)
            the bins are constructed based on ``sfreq`` and ``max_lag``.
        gauss_fwhm : float | None
            Full-width at half maximum of the gaussian kernel to convolve the
            cross-correlation histograms with. Defaults to ``None`` which ommits
            convolution.
        backend : str
            Backend to use for the computation. Can be ``'numpy'``, ``'numba'`` or
            ``'auto'``. Defaults to ``'auto'`` which will use ``'numba'`` if
            numba is available (and number of spikes is > 1000 in any of the cells
            picked) and ``'numpy'`` otherwise.

        Returns
        -------
        xcorr : xarray.DataArray
            Xarray DataArray of cross-correlation histograms. The first
            dimension is the cell pair, and the last dimension is correlation
            the lag. If the input is SpikeEpochs then the second dimension is
            the trial.
        """
        return xcorr_hist(
            self, picks=picks, picks2=picks2, sfreq=sfreq, max_lag=max_lag,
            bins=bins, gauss_fwhm=gauss_fwhm, backend=backend
        )


def _check_waveforms(times, waveform, waveform_time):
    '''Safety checks for waveform data.'''
    n_waveforms = len(waveform)
    assert len(times) == n_waveforms

    ignore_waveforms = [x is None for x in waveform]
    n_spikes_times = np.array(
        [len(x) for x, ign in zip(times, ignore_waveforms) if not ign])
    n_spikes_waveform = np.array(
        [x.shape[0] for x, ign in zip(waveform, ignore_waveforms) if not ign])
    assert (n_spikes_times == n_spikes_waveform).all()

    if waveform_time is not None:
        n_times = len(waveform_time)
        check_waveforms = np.where(~np.array(ignore_waveforms))[0]
        n_times_wvfm_arr = [waveform[ix].shape[1]
                            for ix in check_waveforms]

        if n_waveforms > 1:
            msg = ('If `waveform_time` is passed, waveforms for each unit '
                   'need to have the same number of samples (second dimension'
                   ').')
            assert all([n_times_wvfm_arr[0] == x
                        for x in n_times_wvfm_arr[1:]]), msg

        if not n_times == n_times_wvfm_arr[0]:
            msg = ('Length of `waveform_times` and the second dimension of '
                   '`waveform` must be equal.')
            raise RuntimeError(msg)


def _pick_cells(spk, picks=None, query=None):
    '''Select cells by name or index.  Operates in-place.

    Parameters
    ----------
    spk : Spikes | SpikeEpochs
        Spikes object to select cells from.
    picks : int | str | listlike of int | list of str | None
        Cell names or indices to select.
    query : str | None
        Query for ``.cellinfo`` - to pick cells by their properties, not
        names or indices. Used only when ``picks`` is ``None``.
    '''
    if picks is None and query is None:
        return spk

    if picks is None and query is not None:
        assert spk.cellinfo is not None
        cellinfo_sel = spk.cellinfo.query(query)
        picks = cellinfo_sel.index.values
    else:
        picks = _deal_with_picks(spk, picks)

    spk.cell_names = spk.cell_names[picks].copy()

    if isinstance(spk, Spikes):
        spk.timestamps = [spk.timestamps[ix] for ix in picks]
    elif isinstance(spk, SpikeEpochs):
        spk.time = [spk.time[ix] for ix in picks]
        spk.trial = [spk.trial[ix] for ix in picks]

        if spk.timestamps is not None:
            spk.timestamps = [spk.timestamps[ix] for ix in picks]

    if spk.cellinfo is not None:
        spk.cellinfo = spk.cellinfo.loc[picks, :].reset_index(drop=True)
    if spk.waveform is not None:
        spk.waveform = [spk.waveform[ix] for ix in picks]

    return spk


def _drop_cells(spk, picks):
    '''Drop cells by index. Operates in-place.

    Parameters
    ----------
    picks : int | str | listlike of int
        Cell  indices to drop.
    '''
    all_idx = np.arange(spk.n_units())
    picks = _deal_with_picks(spk, picks)
    is_dropped = np.in1d(all_idx, picks)
    retain_idx = np.where(~is_dropped)[0]
    return spk.pick_cells(retain_idx)


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

    # make sure that cellinfo is present
    if spk.cellinfo is None:
        raise ValueError('To sort units .cellinfo attribute has to contain '
                         'a dataframe with information about the units.')

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

        # FIX: this could be changed into using np.unique() on spk.trial
        frate = compute_spike_rate(spk, step=False, tmin=tmin, tmax=tmax)
        n_spk = (frate.values * winlen).astype('int')
        return n_spk
