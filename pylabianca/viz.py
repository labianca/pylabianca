import numpy as np
import matplotlib.pyplot as plt


# TODO - ! x_dim='auto' (infers which is the likely x dimension) !
# TODO - ! also support mask !
# TODO - allow for colors (use ``mpl.colors.to_rgb('C1')`` etc.)
# TODO - get y axis from xarray data name (?)
# TODO - the info about "one other dimension" (that is reduced) seems to be no
#        longer accurate
def plot_spike_rate(frate, reduce_dim='trial', groupby=None, ax=None,
                    x_dim='time'):
    '''Plot spike rate with standard error of the mean.

    Parameters
    ----------
    frate : xarray.DataArray
        Xarray with ``'time'`` and one other dimension. The other dimension
        is reduced (see ``reduce_dim`` argument below).
    reduce_dim : str
        The dimension to reduce (average). The standard error is also computed
        along this dimension. The default is ``'trial'``.
    groupby : str | None
        The dimension (or sub-dimension) to use as grouping variable plotting
        the spike rate into separate lines. The default is ``None``, which
        does not perform grouping.
    ax : matplotlib.Axes | None
        Axis to plot into. The default is ``None`` which creates an new axis.

    Returns
    -------
    ax : matplotlib.Axes
        Axis with the plot.
    '''
    if ax is None:
        _, ax = plt.subplots()

    if ('cell' in frate.coords and not reduce_dim == 'cell'
        and len(frate.cell.shape) > 0):
        if len(frate.coords['cell'] == 1):
            frate = frate.isel(cell=0)
        else:
            msg = ('DataArray contains more than one cell to plot - this is '
                    'not supported.')
            raise RuntimeError(msg)

    # compute mean, std and n
    if groupby is not None:
        frate = frate.groupby(groupby)
    avg = frate.mean(dim=reduce_dim)
    std = frate.std(dim=reduce_dim)
    n = frate.count(dim=reduce_dim)

    # calculate standard error of the mean
    std_err = std / np.sqrt(n)
    ci_low = avg - std_err
    ci_high = avg + std_err

    # plot each line with error interval
    if groupby is not None:
        sel = {groupby: 0}
        for val in avg.coords[groupby]:
            sel[groupby] = val.item()
            lines = avg.sel(**sel).plot(label=val.item(), ax=ax)
            ax.fill_between(avg.coords[x_dim], ci_low.sel(**sel),
                            ci_high.sel(**sel), alpha=0.3)
    else:
        lines = avg.plot(ax=ax)
        ax.fill_between(avg.coords[x_dim], ci_low, ci_high, alpha=0.3)

    if x_dim == 'time':
        ax.set_xlabel('Time (s)', fontsize=14)

    ax.set_ylabel('Spike rate (Hz)', fontsize=14)
    if groupby is not None:
        ax.legend(title=f'{groupby}:')

    add_txt = '' if groupby is None else f' grouped by {groupby}'
    ttl = 'Firing rate' + add_txt
    ax.set_title(ttl, fontsize=16)
    return lines[0].axes


# TODO: move to sarna sometime
# TODO: make sure the pbar is cleared ... (tqdm._instances.clear() may help)
def check_modify_progressbar(pbar, total=None):
    '''Reset ``pbar`` and change its total if it is a tqdm progressbar.
    Otherwise create new progressbar.

    Parameters
    ----------
    pbar : bool | str | tqdm progressbar
        Progressbar to modify or instructions on whether (if bool) or of what
        kind (if str) a progressbar should be created.
    total : int | None
        Total number of steps in the progressbar.

    Returns
    -------
    pbar : tqdm progressbar | empty sarna progressbar
        Modified or newly created progressbar. Can also be an empty progressbar
        from sarna - a progressbar that has some of the tqdm's API but does not
        do anything.
    '''
    if isinstance(pbar, bool):
        if pbar:
            from tqdm.auto import tqdm
            pbar = tqdm(total=total)
        else:
            from sarna.utils import EmptyProgressbar
            pbar = EmptyProgressbar(total=total)
    elif isinstance(pbar, str):
        if pbar == 'notebook':
            from tqdm.notebook import tqdm
        elif pbar == 'text':
            from tqdm import tqdm
        pbar = tqdm(total=total)
    else:
        from tqdm.notebook import tqdm_notebook
        if isinstance(pbar, tqdm_notebook):
            pbar.reset(total=total)
    return pbar


# TODO:
# - [ ] ! fix x axis units !
# - [ ] kind='line' ?
# - [ ] datashader backend
# - [ ] allow to plot multiple average waveforms as lines
def plot_waveform(spk, pick=0, upsample=False, ax=None, labels=True):
    '''Plot waveform heatmap for one cell.

    Parameters
    ----------
    spk : pylabianca.spikes.Spikes | pylabianca.spikes.SpikeEpochs
        Spike object to use.
    pick : int | str
        Cell index to plot waveform for.
    upsample : bool | float
        Whether to upsample the waveform (defaults to ``False``). If
        ``True`` the waveform is upsampled by a factor of three. Can also
        be a value to specify the upsampling factor.
    ax : matplotlib.Axes | None
        Axis to plot to. By default opens a new figure.
    labels : bool
        Whether to add labels to the axes.
    y_bins : int
        How many bins to use for the y axis.

    Returns
    -------
    ax : matplotlib.Axes
        Axis with the waveform plot.
    '''
    from pylabianca.utils import _deal_with_picks

    pick = _deal_with_picks(spk, pick)[0]
    n_spikes, n_samples = spk.waveform[pick].shape
    waveform = spk.waveform[pick]
    if upsample:
        if isinstance(upsample, bool):
            upsample = 3
        from scipy import interpolate

        x = np.arange(n_samples)
        interp = interpolate.interp1d(x, waveform, kind='cubic',
                                      assume_sorted=True)

        new_x = np.linspace(0, n_samples - 1, num=n_samples * upsample)
        waveform = interp(new_x)
        n_samples = len(new_x)
    else:
        upsample = 1

    x_coords = np.tile(np.arange(n_samples), (n_spikes, 1))

    # assume default combinato times
    sfreq = 32_000
    sample_time = 1000 / sfreq
    time_edges = [-19 * sample_time,
                  (n_samples / upsample - 20) * sample_time]

    hist, xbins, ybins = np.histogram2d(x_coords.ravel(),
                                        waveform.ravel(),
                                        bins=[n_samples, 100])

    max_alpha = np.percentile(hist[hist > 0], 45)
    max_lim = np.percentile(hist[hist > 0], 99)

    alpha2 = hist.T * (hist.T <= max_alpha) / max_alpha
    alpha_sum = (hist.T > max_alpha).astype('float') + alpha2
    alpha_sum[alpha_sum > 1] = 1

    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(hist.T, alpha=alpha_sum, vmax=max_lim, origin='lower',
              extent=(time_edges[0], time_edges[-1], ybins[0], ybins[-1]),
              aspect='auto')
    if labels:
        ax.set_xlabel('Time (ms)', fontsize=14)
        ax.set_ylabel('Amplitude ($\mu$V)', fontsize=14)
    return ax


# TODO: add order=False for groupby?
def plot_raster(spk, pick=0, groupby=None, ax=None):
    '''Show spike rasterplot.

    Parameters
    ----------
    spk : pylabianca.spikes.Spikes | pylabianca.spikes.SpikeEpochs
        Spike object to use.
    pick : int | str
        Cell index or name to plot raster for.
    groupby : str | None
        If not None, group the raster by given variable (requires present
        ``.metadata`` field of the ``spk``).

    Returns
    -------
    ax : matplotlib.Axes
        Axis with the raster plot.
    '''

    if ax is None:
        _, ax = plt.subplots()

    spk_cell = spk.copy().pick_cells(picks=pick)

    tri_spikes = list()
    colors = list()

    if groupby is not None:
        values = np.unique(spk_cell.metadata.loc[:, groupby])
    else:
        values = [None]

    for idx, value in enumerate(values):
        img_color = f'C{idx}'
        if groupby is not None:
            trials = (spk_cell.metadata.query(f'{groupby} == {value}')
                      .index.values)
        else:
            trials = spk_cell.metadata.index.values

        for trial in trials:
            msk = spk_cell.trial[0] == trial
            tri_spikes.append(spk_cell.time[0][msk])
            colors.append(img_color)

    ax.eventplot(tri_spikes, colors=colors)

    # set y limits
    n_trials = len(tri_spikes)
    ax.set_ylim(-1, n_trials)

    # set x limits
    xlim = spk.time_limits + np.array([-0.05, 0.05])
    ax.set_xlim(xlim)

    return ax


def plot_spikes(spk, frate, groupby=None, df_clst=None, pick=0,
                min_pval=0.001):
    '''PLot average spike rate and spike raster.

    spk : pylabianca.spikes.SpikeEpochs
        Spike object to use.
    frate : xarray.DataArray
        Firing rate xarray in the format returned by ``spk.spike_rate()`` or
        ``spk.spike_density()``.
    groupby : str | None
        How to group the plots. If None, no grouping is done.
    df_clst : pandas.DataFrame | None
        DataFrame with cluster time ranges and p values. If None, no cluster
        information is shown.
    pick : int | str
        Name or index of the cell to plot.
    min_pval : float
        Minimum p-value of cluster to mark on the plot. Only used if
        ``df_clst`` is not None.

    Returns
    -------
    fig : matplotlib.Figure
        Figure with the plots.
    '''
    # select cell from frate
    if isinstance(pick, str):
        cell_name = pick
        this_frate = frate.sel(cell=cell_name)
    else:
        this_frate = frate.isel(cell=pick)
        cell_name = this_frate.coords['cell'].item()

    # plot
    gridspec_kw = {'bottom': 0.15, 'left': 0.15}
    fig, ax = plt.subplots(nrows=2, gridspec_kw=gridspec_kw)
    plot_spike_rate(this_frate, groupby=groupby, ax=ax[0])

    # add highlight
    if df_clst is not None:
        this_clst = df_clst.query(f'neuron == "{cell_name}"')
        pvals = this_clst.pval.values
        masks = [_create_mask_from_window_str(twin, this_frate)
                 for twin in this_clst.window.values]
        add_highlights(this_frate, masks, pvals, ax=ax[0],
                       min_pval=min_pval)

    plot_raster(spk.copy().pick_cells(cell_name), pick=0,
                groupby=groupby, ax=ax[1])
    ylim = ax[1].get_xlim()
    ax[0].set_xlim(ylim)

    ax[0].set_xlabel('')
    ax[0].set_ylabel('Spike rate (Hz)', fontsize=12)
    ax[1].set_xlabel('Time (s)', fontsize=12)
    ax[1].set_ylabel('Trials', fontsize=12)

    return fig


def _create_mask_from_window_str(window, frate):
    twin = [float(x) for x in window.split(' - ')]
    mask = (frate.time.values >= twin[0]) & (frate.time.values <= twin[1])
    return mask


def add_highlights(arr, clusters, pvals, p_threshold=0.05, ax=None,
                   min_pval=0.001):
    '''FIXME: add docstring.'''
    import sarna

    if ax is None:
        ax = plt.gca()

    if pvals is None:
        return
    pvals_significant = pvals < p_threshold
    last_dim = arr.dims[-1]

    if pvals_significant.any():
        import borsar

        ylm = ax.get_ylim()
        y_rng = np.diff(ylm)[0]
        text_y = ylm[1] - 0.01 * y_rng
        ax.set_ylim([ylm[0], ylm[1] + 0.1 * y_rng])

        sig_idx = np.where(pvals_significant)[0]
        x_coords = arr.coords[last_dim].values

        sig_clusters = [clusters[ix] for ix in sig_idx]
        sarna.viz.highlight(x_coords, sig_clusters,
                            bottom_bar=True, axis=ax)

        for ix in sig_idx:
            this_pval = pvals[ix]
            text_x = x_coords[clusters[ix]][0]

            if this_pval < min_pval:
                p_txt = 'p < {:.3f}'.format(min_pval)
            else:
                p_txt = borsar.stats.format_pvalue(this_pval)

            ax.text(text_x, text_y, p_txt)


def align_axes_limits(axes=None, ylim=True, xlim=False):
    '''Align the limits of all ``axes``.'''
    if axes is None:
        axes = plt.gcf().get_axes()

    do_lim = dict(x=xlim, y=ylim)
    limits = dict(x=[np.inf, -np.inf], y=[np.inf, -np.inf])
    iter = (axes if isinstance(axes, (list, np.ndarray))
            else axes.values() if isinstance(axes, dict) else None)

    for ax in iter:
        get_lim = dict(x=ax.get_xlim, y=ax.get_ylim)
        for lim in ('x', 'y'):
            if do_lim[lim]:
                this_lim = get_lim[lim]()
                if limits[lim][0] > this_lim[0]:
                    limits[lim][0] = this_lim[0]
                if limits[lim][1] < this_lim[1]:
                    limits[lim][1] = this_lim[1]

    iter = (axes if isinstance(axes, (list, np.ndarray))
            else axes.values() if isinstance(axes, dict) else None)
    for ax in iter:
        set_lim = dict(x=ax.set_xlim, y=ax.set_ylim)
        for lim in ('x', 'y'):
            if do_lim[lim]:
                set_lim[lim](limits[lim])
