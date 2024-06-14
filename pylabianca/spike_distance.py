import itertools
import numpy as np

from . import utils, viz
from .utils import (_deal_with_picks, _turn_spike_rate_to_xarray,
                    _symmetric_window_samples, _gauss_kernel_samples,
                    spike_centered_windows, has_numba)


# TODO: ``tol=None`` could return the distances without thresholding
# TODO: consider other name like ``find_spike_distances``
# TODO: test for presence of numba and use the numba implementation
#       (could also use ``backend`` argument)
# TODO: multiprocessing could be useful
# TODO: the implementation and the API are suboptimal
# TODO: remove the numpy backend?
def compare_spike_times(spk, cell_idx1, cell_idx2, spk2=None, backend='numba',
                        tol=None):
    '''Test concurrence of spike times for Spikes or SpikeEpochs.

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
    backend : str
        Backend to use for the computation. Currently only 'numba' is
        supported.
    tol : float
        Concurrence tolerance in seconds. Spikes no further that this will be
        deemed co-occurring. Default is ``None``, which means no concurrence
        threshold will be applied and a distance matrix will be returned.

    Returns
    -------
    distance: float | array of float
        If ``tol`` is ``None`` an array of time distances between closest
        spikes of the two cells is returned. More precisely, for each spike
        from cell 1 a distance to closest cell 2 spike is given. If ``tol`` is
        not ``None`` then a percentage of spikes from first cell that concur
        with spikes from the second cell is given.
    '''
    from .spikes import SpikeEpochs, Spikes

    if isinstance(spk, SpikeEpochs):
        raise NotImplementedError('Sorry compare_spike_times does not work '
                                  'with SpikeEpochs yet.')
    elif isinstance(spk, Spikes):
        if backend == 'numba':
            from ._numba import numba_compare_times
            distances = numba_compare_times(spk, cell_idx1, cell_idx2,
                                            spk2=spk2)
            if tol is not None:
                distances = (distances < tol).mean()
            return distances
        else:
            if spk2 is not None:
                raise NotImplementedError(
                    'Sorry compare_spike_times does not support spk2 '
                    'in the numpy backend.'
                )

            # TODO: rework this
            tms1 = spk.timestamps[cell_idx1] / spk.sfreq
            tms2 = spk.timestamps[cell_idx2] / spk.sfreq
            time_diffs = np.abs(tms1[:, None] - tms2[None, :])
            closest_time1 = time_diffs.min(axis=1)
            return (closest_time1 < tol).mean()


def numpy_compare_times(spk, cell_idx1, cell_idx2):
    tms1 = spk.timestamps[cell_idx1] / spk.sfreq
    tms2 = spk.timestamps[cell_idx2] / spk.sfreq
    time_diffs = np.abs(tms1[:, None] - tms2[None, :])
    closest_time1 = time_diffs.min(axis=1)
    return closest_time1


def compute_spike_coincidence_matrix(spk, spk2=None, tol=0.002, progress=True):
    if progress:
        from tqdm import tqdm

    n_cells = len(spk)
    n_cells2 = n_cells if spk2 is None else len(spk2)

    similarity = np.zeros((n_cells, n_cells2))
    iter_over = tqdm(range(n_cells)) if progress else range(n_cells)

    for cell1 in iter_over:
        for cell2 in range(n_cells2):
            if spk2 is None and cell1 == cell2:
                continue

            simil = compare_spike_times(spk, cell1, cell2, spk2=spk2, tol=tol)
            similarity[cell1, cell2] = simil

    return similarity


def _construct_bins(sfreq, max_lag, bins=None):
    '''Construct bins for cross-correlation histogram.

    Parameters
    ----------
    sfreq : float
        Sampling frequency of the bins. The bin width will be ``1 / sfreq``
        seconds. Used only when ``bins is None``.
    max_lag : float
        Maximum lag in seconds. Used only when ``bins is None``.
    bins : numpy array | None
        Array representing edges of the histogram bins. If ``None`` (default)
        the bins are constructed based on ``sfreq`` and ``max_lag``.

    Returns
    -------
    bins : numpy array
        Array representing edges of the histogram bins.
    '''
    if bins is not None:
        return bins
    else:
        smp_time = 1 / sfreq
        bins = np.arange(-max_lag, max_lag + smp_time / 2, step=smp_time)

        # check if middle bin is zero-centered
        bin_width = np.diff(bins[:2])[0]
        mid_bin = np.abs(bins).argmin()
        diff_to_centered = np.abs(bins[mid_bin] - (bin_width / 2))

        if diff_to_centered > (bin_width / 10):
            bins = bins - diff_to_centered
            bins = np.concatenate([bins, [bins[-1] + bin_width]])

    return bins


# TODO: add option to shuffle trials?
# TODO: add option to shift trials?
# TODO: split into a higher and lower level function
#       the higher level would be common to this and _xcorr_hist
def _xcorr_hist_trials(spk, cell_idx1, cell_idx2, sfreq=500., max_lag=0.2,
                       bins=None):
    '''
    Compute cross-correlation histogram between two cells of SpikeEpochs.

    Parameters
    ----------
    spk : pylabianca.spikes.SpikeEpochs
        SpikeEpochs object to use.
    cell_idx1 : int
        Index of the first cell to compute cross-correlation for.
    cell_idx2 : int
        Index of the second cell to compute cross-correlation for. If
        ``cell_idx1 == cell_idx2`` then auto-correlation will be computed.
    sfreq : float
        Sampling frequency of the bins. The bin width will be ``1 / sfreq``
        seconds. Used only when ``bins is None``. Defaults to ``500.``.
    max_lag : float
        Maximum lag in seconds. Used only when ``bins is None``. Defaults
        to ``0.2``.
    bins : numpy array | None
        Array representing edges of the histogram bins. If ``None`` (default)
        the bins are constructed based on ``sfreq`` and ``max_lag``.

    Returns
    -------
    bins : numpy array
        Array representing edges of the histogram bins.
    '''
    from .utils import _get_trial_boundaries

    bins = _construct_bins(sfreq, max_lag, bins=bins)
    max_tri = (spk.n_trials if spk.n_trials is not None
               else max([tri.max() for tri in spk.trial]))
    xcorr = np.zeros((max_tri + 1, len(bins) - 1), dtype=int)

    # TODO: this could be done once at the beginning
    trial_boundaries1, tri_num1 = _get_trial_boundaries(spk, cell_idx1)
    if cell_idx1 == cell_idx2:
        autocorr = True
        trial_boundaries2, tri_num2 = trial_boundaries1, tri_num1
    else:
        autocorr = False
        trial_boundaries2, tri_num2 = _get_trial_boundaries(spk, cell_idx2)

    idx2 = 0
    for idx1, tri in enumerate(tri_num1):

        if not autocorr:
            # check if the other unit has spikes in this trial
            has_tri = tri in tri_num2[idx2:]
            if has_tri:
                idx2 = np.where(tri_num2[idx2:] == tri)[0][0] + idx2
            else:
                continue

        times1 = spk.time[cell_idx1][
            trial_boundaries1[idx1]:trial_boundaries1[idx1 + 1]]

        if autocorr:
            times2 = times1
        else:
            times2 = spk.time[cell_idx2][
                trial_boundaries2[idx2]:trial_boundaries2[idx2 + 1]]

        time_diffs = times2[:, None] - times1[None, :]

        if autocorr:
            n_diffs = time_diffs.shape[0]
            ind = np.diag_indices(n_diffs)
            time_diffs[ind] = np.nan

        time_diffs = time_diffs.ravel()
        this_hist, _ = np.histogram(time_diffs, bins=bins)
        xcorr[tri, :] = this_hist

    return xcorr, bins


# TODO: max_lag cannot be longer than epoch time window and preferably
#       it is < 1 / 2 epoch length ...
# TODO: when gauss_fwhm is set, calculate trim before and adjust max_lag
#       so that it is left in after convolution
def xcorr_hist(spk, picks=None, picks2=None, sfreq=500., max_lag=0.2,
               bins=None, gauss_fwhm=None, backend='auto'):
    '''Compute cross-correlation histogram for each trial.

    Parameters
    ----------
    spk : Spikes | SpikeEpochs
        SpikeEpochs to compute cross-correlation for.
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
        seconds. Used only when ``bins is None``. Defaults to ``500.``.
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
        Xarray DataArray of cross-correlation histograms. The first dimension
        is the cell pair, and the last dimension is correlation the lag. If the
        input is SpikeEpochs then the second dimension is the trial.
    '''
    from .spikes import Spikes, SpikeEpochs
    from .utils import _deal_with_picks, _turn_spike_rate_to_xarray
    assert isinstance(spk, (Spikes, SpikeEpochs))

    has_epochs = isinstance(spk, SpikeEpochs)
    if not has_epochs:
        spk = spk.to_epochs()
    else:
        has_epochs = spk.n_trials > 1

    pick_pairs, picks, picks2 = _deal_with_pick_pairs(
        spk, picks, picks2=picks2)
    bins = _construct_bins(sfreq, max_lag, bins=bins)

    if backend == 'auto':
        min_spikes_numba = 1_000

        if picks2 is not None:
            all_picks = np.concatenate([picks, picks2])
        else:
            all_picks = picks

        n_spikes = spk.n_spikes()
        n_spikes = n_spikes[all_picks]
        valid_spikes =  (n_spikes > min_spikes_numba).any()
        if valid_spikes and has_numba():
            backend = 'numba'
        else:
            backend = 'numpy'

    if backend == 'numba' and not has_epochs:
        from ._numba import _xcorr_hist_auto_numba, _xcorr_hist_cross_numba
        auto_fun = _xcorr_hist_auto_numba
        cross_fun = _xcorr_hist_cross_numba
    elif backend == 'numpy':
        auto_fun = _xcorr_hist_auto_py
        cross_fun = _xcorr_hist_cross_py
    else:
        raise ValueError(f'Unknown backend: {backend}')

    cell = list()  # pair name, but set as cell attr in xarray
    cell1_idx, cell2_idx = list(), list()
    cell1_name = list()
    cell2_name = list()
    auto = list()  # is autocorrelation

    # LOOP
    xcorrs = list()
    for idx1, idx2  in pick_pairs:
        is_auto = idx1 == idx2

        # compute xcorr histogram for cell pair
        # -------------------------------------
        if has_epochs:
            xcorr, _ = _xcorr_hist_trials(
                spk, idx1, idx2, sfreq=sfreq, max_lag=max_lag, bins=bins)
        else:
            if is_auto:
                xcorr = auto_fun(spk.time[idx1], bins)
            else:
                xcorr = cross_fun(spk.time[idx1], spk.time[idx2], bins)

        # add coords to lists
        name1 = spk.cell_names[idx1]
        name2 = spk.cell_names[idx2]
        name = f'{name1} x {name2}'

        cell.append(name)
        cell1_name.append(name1)
        cell2_name.append(name2)
        cell1_idx.append(idx1)
        cell2_idx.append(idx2)

        xcorrs.append(xcorr)

    # stack cell pairs as first dimension
    xcorrs = np.stack(xcorrs, axis=0)

    # calc bin_centers
    bin_widths = np.diff(bins)
    bin_centers = bins[:-1] + bin_widths

    # smooth with gaussian if needed
    # TODO: DRY with _spike_density
    if gauss_fwhm is not None:
        xcorrs, bin_centers = _convolve_xcorr(
            xcorrs, bin_centers, gauss_fwhm, sfreq)

    # construct xarr
    xcorrs = _turn_spike_rate_to_xarray(
        bin_centers, xcorrs, spk, cell_names=cell, x_dim_name='lag')
    xcorrs.name = 'count'
    xcorrs.attrs['unit'] = 'n'
    xcorrs.attrs['coord_units'] = {'lag': 's'}

    # add cell1_idx etc.
    xcorrs = xcorrs.assign_coords(
        {'cell1_name': ('cell', cell1_name),
         'cell2_name': ('cell', cell2_name),
         'cell1_idx': ('cell', cell1_idx),
         'cell2_idx': ('cell', cell2_idx)}
    )

    return xcorrs


# CONSIDER: move out to utils?
def _convolve_xcorr(xcorrs, bin_centers, gauss_fwhm, sfreq):
    from scipy.signal import correlate
    from .spike_rate import _gauss_sd_from_FWHM

    gauss_sd = _gauss_sd_from_FWHM(gauss_fwhm)
    winlen = gauss_sd * 6
    gauss_sd = gauss_sd * sfreq
    win_smp, trim = _symmetric_window_samples(winlen, sfreq)
    kernel = _gauss_kernel_samples(win_smp, gauss_sd)

    bin_centers = bin_centers[trim:-trim]
    full_kernel = (kernel[None, None, :] if xcorrs.ndim == 3
                   else kernel[None, :] if xcorrs.ndim == 2
                   else kernel)
    xcorrs = correlate(xcorrs, full_kernel, mode='valid')
    return xcorrs, bin_centers


def _deal_with_pick_pairs(spk, picks, picks2=None):
    picks = _deal_with_picks(spk, picks)
    if picks2 is None:
        # picks and picks2 has to be all combinations of picks
        pick_pairs = itertools.product(picks, picks)
    else:
        picks2 = _deal_with_picks(spk, picks2)
        if len(picks) == len(picks2):
            pick_pairs = zip(picks, picks2)
        else:
            pick_pairs = itertools.product(picks, picks2)
    return pick_pairs, picks, picks2


# more memory efficient version of xcorr_hist_trials
# intended to work on Spikes turned to SpikeEpochs
# (_xcorr_hist_trials is VERY memory inefficient for many thousands of spikes
#  but that does not happen frequently with single trials)
# CONSIDER: use batch-array looping instead of full iterative loop
def _xcorr_hist_auto_py(times, bins, batch_size=1_000):
    '''Compute auto-correlation histogram for a single cell.

    [a little more about memory efficiency and using monotonic relationship to
     our advantage etc.]'''
    n_times = times.shape[0]
    distances = list()
    n_bins = len(bins) - 1
    max_lag = max(np.abs(bins[[0, -1]]))
    counts = np.zeros(n_bins, dtype=int)

    in_batch = 0
    for idx1 in range(n_times):
        time1 = times[idx1]

        # move forward till we fall out of max_lag
        max_lag_ok = True
        idx2 = idx1
        while max_lag_ok and idx2 < (n_times - 1):
            idx2 += 1
            distance = times[idx2] - time1
            max_lag_ok = distance <= max_lag

            if max_lag_ok:
                distances.append(distance)
                distances.append(-distance)
                in_batch += 2

        if in_batch >= batch_size or idx1 == (n_times - 1):
            these_counts, _ = np.histogram(distances, bins)
            counts += these_counts
            in_batch = 0
            distances = list()

    return counts


def _xcorr_hist_cross_py(times, times2, bins, batch_size=1_000):
    '''Compute cross-correlation histogram for a single cell.

    [a little more about memory efficiency and using monotonic relationship to
     our advantage etc.]'''
    n_times = times.shape[0]
    n_times2 = times2.shape[0]
    max_lag = max(np.abs(bins[[0, -1]]))
    distances = list()
    n_bins = len(bins) - 1
    counts = np.zeros(n_bins, dtype=int)

    tm_idx_low = 0
    tm_idx_high = 1
    in_batch = 0

    for idx1 in range(n_times):
        time1 = times[idx1]

        # move forward till tm_idx_high
        new_tm_idx_low = tm_idx_low
        for idx2 in range(tm_idx_low, tm_idx_high):
            if not idx1 == idx2:
                distance = times2[idx2] - time1

                if distance < -max_lag:
                    new_tm_idx_low = idx2
                else:
                    distances.append(distance)
                    in_batch += 1

        tm_idx_low = new_tm_idx_low
        max_lag_ok = True
        while max_lag_ok and idx2 < (n_times2 - 1):
            idx2 += 1
            if not idx1 == idx2:
                distance = times2[idx2] - time1
                max_lag_ok = distance <= max_lag

                if max_lag_ok:
                    tm_idx_high = idx2
                    distances.append(distance)
                    in_batch += 1

        if in_batch >= batch_size or idx1 == (n_times - 1):
            these_counts, _ = np.histogram(distances, bins)
            counts += these_counts
            in_batch = 0
            distances = list()

    return counts
