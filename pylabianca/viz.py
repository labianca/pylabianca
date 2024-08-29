import numpy as np


# TODO - title is now removed, so for groupby it would be good to specify the
#        groupby coord name in legend "title"
# TODO - allow for colors (use ``mpl.colors.to_rgb('C1')`` etc.)
def plot_shaded(arr, reduce_dim=None, groupby=None, ax=None,
                x_dim=None, legend=True, legend_pos=None, colors=None,
                labels=True, **kwargs):
    '''Plot spike rate with standard error of the mean.

    Parameters
    ----------
    frate : xarray.DataArray
        Xarray with at least two dimensions: one is plotted along the x axis
        (this is controlled with ``x_dim`` argument); the other is reduced
        by averaging (see ``reduce_dim`` argument below). The averaged
        dimension also gives rise to the standard error of the mean, which is
        plotted as a shaded area.
    reduce_dim : str
        The dimension to reduce (average). The standard error is also computed
        along this dimension. If ``reduce_dim`` is ``None`` (default) then the
        following dimensions are tested in this order: ``['trial', 'fold',
        'perm', 'permutation', 'cell', 'spike']``. If none of these dimensions
        are found, the first dimension is reduced.
    groupby : str | None
        The dimension (or sub-dimension) to use as grouping variable plotting
        the spike rate into separate lines. The default is ``None``, which
        does not perform grouping.
    ax : matplotlib.Axes | None
        Axis to plot into. The default is ``None`` which creates an new axis.
    x_dim : str
        Dimension to use for the x axis. The default is ``'time'``.
    legend : bool
        Whether to plot the legend.
    legend_pos : str | None
        Legend position (standard matplotlib names like "upper left"). Defaults
        to ``None`` which uses ``'best'`` position.
    colors : list of arrays | dictionary of arrays | None
        List of RGB arrays to use as colors for condition groups. Can also be
        a dictionary linking condition names / values and RBG arrays. Default
        is ``None`` which uses the default matplotlib color cycle.
    labels : bool
        Whether to add labels to the axes.

    Returns
    -------
    ax : matplotlib.Axes
        Axis with the plot.
    '''
    # auto-infer reduce_dim
    # ---------------------
    if reduce_dim is None:
        auto_reduce_dims = ['trial', 'fold', 'perm', 'permutation', 'cell',
                            'spike']
        if groupby is not None and groupby in auto_reduce_dims:
            auto_reduce_dims.remove(groupby)

        for dim_name in auto_reduce_dims:
            if dim_name in arr.dims:
                reduce_dim = dim_name
                break

    # if reduce_dim is still None - use the first dim for 2d array
    if reduce_dim is None:
        reduce_dim = arr.dims[0]

    # make sure enough dimensions to plot
    plot_ndim = arr.ndim - 1  # -1 for reduce_dim
    if groupby is not None and groupby in arr.dims:
        plot_ndim -= 1

    if plot_ndim > 1:
        msg = ('DataArray contains too many dimensions to plot. The number of'
               f'dimensions in the array is {arr.ndim}, while {plot_ndim} '
               'would have to be plotted (after accounting for the reduced '
               'dimension and the groupby dimension).')
        raise ValueError(msg)

    # auto-infer x_dim
    # ----------------
    # try time, freq, frequency, lag by default
    if x_dim is None:
        auto_x_dims = ['time', 'frequency', 'freq', 'lag']
        for dim_name in auto_x_dims:
            if dim_name in arr.coords:
                x_dim = dim_name
                break

    # if x_dim is still None - use the last dim for 2d array
    if x_dim is None:
        x_dim = arr.dims[-1]

    ax = plot_xarray_shaded(
        arr, reduce_dim=reduce_dim, x_dim=x_dim, groupby=groupby, ax=ax,
        legend=legend, legend_pos=legend_pos, colors=colors, **kwargs
    )

    # clean up ax title if groupby is used
    if groupby is not None:
        title = ax.get_title()
        if groupby in title and ', ' in title:
            title = title.split(', ')
            title = [x for x in title if not x.startswith(groupby)]
            title = ', '.join(title)
            ax.set_title(title)

    if labels:
        xlabel = x_dim.capitalize()
        dim_unit = x_dim + '_unit'
        if dim_unit in arr.attrs:
            this_unit = arr.attrs[dim_unit]
            xlabel += f' ({this_unit})'
        ax.set_xlabel(xlabel, fontsize=14)

        if arr.name is not None:
            ylabel = arr.name.capitalize()
            if 'unit' in arr.attrs:
                this_unit = arr.attrs['unit']
                ylabel += f' ({this_unit})'
        else:
            ylabel=''
        ax.set_ylabel(ylabel, fontsize=14)

    return ax


# TODO: allow different error shades
def plot_xarray_shaded(arr, reduce_dim=None, x_dim='time', groupby=None,
                       ax=None, legend=True, legend_pos=None, colors=None,
                       **kwargs):
    """
    arr : xarray.DataArray
        Xarray with at least two dimensions: one is plotted along the x axis
        (this is controlled with ``x_dim`` argument); the other is reduced
        by averaging (see ``reduce_dim`` argument below). The averaged
        dimension also gives rise to the standard error of the mean, which is
        plotted as a shaded area.
    reduce_dim : str
        The dimension to reduce (average). The standard error is also computed
        along this dimension. The default is ``'trial'``.
    x_dim : str
        Dimension to use for the x axis. The default is ``'time'``.
    groupby : str | None
        The dimension (or sub-dimension) to use as grouping variable plotting
        the spike rate into separate lines. The default is ``None``, which
        does not perform grouping.
    ax : matplotlib.Axes | None
        Axis to plot into. The default is ``None`` which creates an new axis.
    legend : bool
        Whether to plot the legend.
    legend_pos : str | None
        Legend position (standard matplotlib names like "upper left"). Defaults
        to ``None`` which uses ``'best'`` position.
    colors : list of arrays | dictionary of arrays | None
        List of RGB arrays to use as colors for condition groups. Can also be
        a dictionary linking condition names / values and RBG arrays. Default
        is ``None`` which uses the default matplotlib color cycle.
    """
    import matplotlib.pyplot as plt
    assert reduce_dim is not None

    if ax is None:
        _, ax = plt.subplots()

    # compute mean, std and n
    if groupby is not None:
        arr = arr.groupby(groupby)

    # calculate standard error of the mean
    avg = arr.mean(dim=reduce_dim)
    std = arr.std(dim=reduce_dim)
    n = arr.count(dim=reduce_dim)
    std_err = std / np.sqrt(n)
    ci_low = avg - std_err
    ci_high = avg + std_err

    # handle colors
    if colors is not None:
        if groupby is not None:
            group_names = avg.coords[groupby].values
            n_groups = len(group_names)
        else:
            n_groups = 1
            group_names = ['base']

            if len(colors) > 1:
                colors = [colors]

        assert len(colors) == n_groups
        if isinstance(colors, (list, tuple, np.ndarray)):
            assert all(isinstance(x, (list, tuple, np.ndarray)) for x in colors)
            colors = {group: color for group, color in zip(group_names, colors)}
        else:
            assert all(name in colors.keys() for name in group_names)
            assert all(isinstance(x, (list, tuple, np.ndarray))
                       for x in colors.values())

    # plot each line with error interval
    if groupby is not None:
        sel = {groupby: 0}
        for val in avg.coords[groupby]:
            val = val.item()
            sel[groupby] = val

            add_arg = {'color': colors[val]} if colors is not None else dict()
            if len(kwargs) > 0:
                line_args = add_arg.copy()
                line_args.update(kwargs)
                lines = avg.sel(**sel).plot(label=val, ax=ax, **line_args)
            else:
                lines = avg.sel(**sel).plot(label=val, ax=ax, **add_arg)
            ax.fill_between(avg.coords[x_dim], ci_low.sel(**sel),
                            ci_high.sel(**sel), alpha=0.3, linewidth=0,
                            **add_arg)
    else:
        add_arg = {'color': colors['base']} if colors is not None else dict()
        if len(kwargs) > 0:
            line_args = add_arg.copy()
            line_args.update(kwargs)
            lines = avg.plot(ax=ax, **line_args)
        else:
            lines = avg.plot(ax=ax, **add_arg)
        ax.fill_between(avg.coords[x_dim], ci_low, ci_high, linewidth=0,
                        alpha=0.3, **add_arg)

    if groupby is not None and legend:
        pos = 'best' if legend_pos is None else legend_pos
        ax.legend(title=f'{groupby}:', loc=pos)

    return lines[0].axes


# TODO: move to borsar sometime
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
            from tqdm import tqdm
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
        try:
            from tqdm import tqdm
            if isinstance(pbar, tqdm):
                pbar.reset(total=total)
            else:
                from tqdm.notebook import tqdm_notebook
                if isinstance(pbar, tqdm_notebook):
                    pbar.reset(total=total)
        except ImportError:
            pbar = EmptyProgressbar(total=total)
    return pbar


class EmptyProgressbar(object):
    def __init__(self, total=None):
        self.total = total

    def update(self, val):
        pass


# TODO:
# - [ ] kind='line' ?
# - [x] datashader backend?
# - [ ] allow to plot multiple average waveforms as lines
def plot_waveform(spk, picks=None, upsample=False, ax=None, labels=True,
                  n_bins_y=100, times=None, cmap='viridis', backend='numpy'):
    '''Plot waveform heatmap for one cell.

    Parameters
    ----------
    spk : pylabianca.spikes.Spikes | pylabianca.spikes.SpikeEpochs
        Spike object to use.
    picks : int | str
        Cell index to plot waveform for.
    upsample : bool | float
        Whether to upsample the waveform (defaults to ``False``). If
        ``True`` the waveform is upsampled by a factor of three. Can also
        be a value to specify the upsampling factor.
    ax : matplotlib.Axes | None
        Axis to plot to. By default opens a new figure.
    labels : bool
        Whether to add labels to the axes.
    n_bins_y : int
        How many bins to use for the y axis. Defaults to 100. Used only in
        the 'numpy' backend.
    times : None | array-like
        If not None, the times to use for the x axis (in milliseconds).
        If None, the x axis is labelled as samples.
    cmap : str
        Colormap to use. Defaults to 'viridis'.
    backend : str
        Computation backend to use. Can be 'numpy' or 'datashader' (defaults
        to 'numpy'). The 'datashader' backend is currently experimental and
        requires the ``datashader`` package.

    Returns
    -------
    ax : matplotlib.Axes
        Axis with the waveform plot.
    '''
    from .utils import _deal_with_picks

    if spk.waveform is None:
        raise RuntimeError('No waveforms are stored in the spike object.')

    picks = _deal_with_picks(spk, picks)
    n_picks = len(picks)
    ax = auto_multipanel(n_picks, ax=ax)
    use_ax = _simplify_axes(ax)

    time_unit = 'samples' if times is None else 'ms'
    upsample = (1 if not upsample else 3 if isinstance(upsample, bool)
                else upsample)

    n_samples = spk.waveform[picks[0]].shape[-1]
    if times is not None:
        time_edges = [times[0], times[-1]]
    else:
        time_edges = [0, n_samples / upsample]
        times = np.linspace(time_edges[0], time_edges[1],
                            num=n_samples * upsample)

    for idx, unit_idx in enumerate(picks):
        this_waveform = spk.waveform[unit_idx]

        if labels:
            use_ax[idx].set_xlabel(f'Time ({time_unit})')
            use_ax[idx].set_ylabel(r'Amplitude ($\mu$V)')
            use_ax[idx].set_title(spk.cell_names[unit_idx])

        if upsample > 1:
            this_waveform = _upsample_waveform(this_waveform, upsample)

        if backend == 'numpy':
            hist, _, ybins = _calculate_waveform_density_image(
                this_waveform, n_bins_y
            )
            alpha_map, vmax = _calculate_alpha_map(hist)

            use_ax[idx].imshow(
                hist.T, alpha=alpha_map, vmax=vmax, origin='lower',
                extent=(time_edges[0], time_edges[-1], ybins[0], ybins[-1]),
                aspect='auto', cmap=cmap
            )
        elif backend == 'datashader':
            img = _draw_waveform_datashader(this_waveform, times,
                                            use_ax[idx], cmap=cmap)
            x_ext = img.coords['x'].values[[0, -1]].tolist()
            y_ext = img.coords['y'].values[[0, -1]].tolist()
            use_ax[idx].imshow(np.array(img.to_pil()), aspect='auto',
                               extent=x_ext + y_ext, interpolation='none')
        else:
            raise ValueError('Backend can be only "numpy" or "datashader".')

    return ax


def _calculate_waveform_density_image(waveform, n_bins_y, density=True,
                                      y_range=None):
    '''Helps in calculating 2d density histogram of the waveforms.

    Parameters
    ----------
    waveform : np.array
        Waveform to plot.
    n_bins_y : int
        How many bins to use for the y axis. Defaults to 100.
    '''

    n_spikes, n_samples = waveform.shape
    x_coords = np.tile(np.arange(n_samples), (n_spikes, 1))

    xs = x_coords.ravel()
    ys = waveform.ravel()
    nan_mask = np.isnan(ys)
    if nan_mask.any():
        range = [[np.nanmin(xs), np.nanmax(xs)],
                 [np.nanmin(ys), np.nanmax(ys)]]
        xs = xs[~nan_mask]
        ys = ys[~nan_mask]
    else:
        range = None

    if y_range is not None:
        if nan_mask.any():
            range[1] = y_range
        else:
            range = [[np.min(xs), np.max(xs)], y_range]

    hist, xbins, ybins = np.histogram2d(
        xs, ys, bins=[n_samples, n_bins_y], range=range, density=density
    )

    return hist, xbins, ybins


def _upsample_waveform(waveform, upsample=3):
    from scipy import interpolate

    n_samples = waveform.shape[-1]
    x = np.arange(n_samples)
    interp = interpolate.interp1d(
        x, waveform, kind='cubic', assume_sorted=True)

    new_x = np.linspace(0, n_samples - 1, num=n_samples * upsample)
    waveform = interp(new_x)
    return waveform


def _calculate_alpha_map(hist, percentile_nontransparent=45,
                         percentile_max=99):
    nonzero_hist = hist[hist > 0]
    max_alpha, vmax = np.percentile(
        nonzero_hist, [percentile_nontransparent, percentile_max])

    alpha2 = hist.T * (hist.T <= max_alpha) / max_alpha
    alpha_map = (hist.T > max_alpha).astype('float') + alpha2
    alpha_map[alpha_map > 1] = 1

    return alpha_map, vmax


def get_axis_size_pix(ax):
    '''Get the size of the axis in pixels.

    Parameters
    ----------
    ax : matplotlib.Axes
        Axis to get the size of.

    Returns
    -------
    size : tuple
        Size of the axis in pixels as ``(width, height)`` tuple.
    '''
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox = ax.get_window_extent()
    width, height = bbox.width, bbox.height
    return width, height


def _draw_waveform_datashader(waveform, waveform_time, ax, cmap='viridis',
                              how='eq_hist'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import datashader as ds
    import datashader.transfer_functions as tf

    n_spikes, n_samples = waveform.shape
    waves = pd.DataFrame(waveform)

    w_pix, h_pix = get_axis_size_pix(ax)
    cvs = ds.Canvas(plot_height=int(h_pix), plot_width=int(w_pix))
    agg = cvs.line(waves, x=waveform_time, y=list(range(n_samples)),
                   agg=ds.count(), axis=1, line_width=0)
    colormap = getattr(plt.cm, cmap)
    img = tf.shade(agg, how=how, cmap=colormap)

    return img


# TODO: add order=False for groupby?
def plot_raster(spk, pick=0, groupby=None, ax=None, colors=None, labels=True,
                legend=True, legend_kwargs=None, eventplot_kwargs=None):
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
    ax : matplotlib.Axes | None
        Matplotlib axis to plot to. If ``None`` a new figure is opened.
    colors : list of str | None
        List of colors to use for each group. If ``None`` uses the default
        matplotlib color cycle.
    labels : bool
        Whether to add labels to the axes.
    legend : bool
        Whether to plot the legend.
    legend_kwargs : dict | None
        Additional keyword arguments for the legend. If ``None`` uses the
        default legend location.
    eventplot_kwargs : dict | None
        Additional keyword arguments for the eventplot.

    Returns
    -------
    ax : matplotlib.Axes
        Axis with the raster plot.
    '''
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    spk_cell = spk.copy().pick_cells(picks=pick)

    tri_spikes = list()
    colors_list = list() if colors is None else colors

    if groupby is not None:
        values = np.unique(spk_cell.metadata.loc[:, groupby])
        string_levels = values.dtype == 'object'
    else:
        values = [None]

    if colors is None:
        n_values = len(values)
        colors = [f'C{idx}' for idx in range(n_values)]
    colors_list = list()

    for idx, value in enumerate(values):
        img_color = colors[idx]
        if groupby is not None:
            query_str = (f'{groupby} == "{value}"' if string_levels
                         else f'{groupby} == {value}')
            trials = (spk_cell.metadata.query(query_str)
                      .index.values)
        else:
            if spk_cell.metadata is not None:
                trials = spk_cell.metadata.index.values
            else:
                trials = np.arange(spk_cell.n_trials)

        for trial in trials:
            msk = spk_cell.trial[0] == trial
            tri_spikes.append(spk_cell.time[0][msk])

            colors_list.append(img_color)

    if eventplot_kwargs is None:
        eventplot_kwargs = dict()

    ax.eventplot(tri_spikes, colors=colors_list, **eventplot_kwargs)

    # set y limits
    n_trials = len(tri_spikes)
    ax.set_ylim(-1, n_trials)

    # set x limits
    xlim = spk.time_limits + np.array([-0.05, 0.05])
    ax.set_xlim(xlim)

    # legend
    if legend:
        from matplotlib.patches import Patch

        patches = [Patch(facecolor=col, label=lab)
                   for col, lab in zip(colors, values)]

        legend_kwargs = (dict(loc='upper left')
                         if legend_kwargs is None else legend_kwargs)
        leg = ax.legend(handles=patches, **legend_kwargs)
        leg.set_title(groupby)

    if labels:
        ax.set_xlabel('Time (s)', fontsize=14)
        ax.set_ylabel('Trial', fontsize=14)

    return ax


def plot_spikes(spk, frate, groupby=None, colors=None, df_clst=None,
                clusters=None, pvals=None, pick=0, p_threshold=0.05,
                min_pval=0.001, ax=None, eventplot_kwargs=None, fontsize=12):
    '''Plot average spike rate and spike raster.

    spk : pylabianca.spikes.SpikeEpochs
        Spike object to use.
    frate : xarray.DataArray
        Firing rate xarray in the format returned by ``spk.spike_rate()`` or
        ``spk.spike_density()``.
    groupby : str | None
        How to group the plots. If None, no grouping is done.
    colors : list of str | None
        List of colors to use for each group. If None uses the default
    df_clst : pandas.DataFrame | None
        DataFrame with cluster time ranges and p values. If None, no cluster
        information is shown. This argument is to support results obtained
        with ``pylabianca.selectivity.cluster_based_selectivity()``, but one
        can also use the more conventional ``clusters``, ``pvals`` and
        ``p_threshold``.
    clusters : list of np.array
        List of boolean arrays, where each array contains cluster membership
        information (which points along the last array dimension contribute
        to the given cluster).
    pvals : list-like
        List or array of cluster p values.
    p_threshold : float
        Alpha significance threshold. Clusters with p value below this
        threshold will be shown.
    pick : int | str
        Name or index of the cell to plot.
    min_pval : float
        Minimum p-value of cluster to mark on the plot. Only used if
        ``df_clst`` is not None.
    ax: matplotlib.Axes | None
        Two axes to plot on: first is used for average firing rate the second
        is used for raster plot. If None, a new figure is created.
    eventplot_kwargs : dict | None
        Additional keyword arguments for the eventplot.
    fontsize : int
        Font size for the labels.

    Returns
    -------
    fig : matplotlib.Figure
        Figure with the plots.
    '''
    import matplotlib.pyplot as plt

    # select cell from frate
    if isinstance(pick, str):
        cell_name = pick
        this_frate = frate.sel(cell=cell_name)
    else:
        this_frate = frate.isel(cell=pick)
        cell_name = this_frate.coords['cell'].item()

    # plot
    if ax is None:
        gridspec_kw = {'bottom': 0.15, 'left': 0.15}
        fig, ax = plt.subplots(nrows=2, gridspec_kw=gridspec_kw)
    else:
        assert(len(ax) == 2)
        fig = ax[0].figure
    plot_shaded(this_frate, groupby=groupby, colors=colors, ax=ax[0])

    # add highlight
    add_highlight = (df_clst is not None) or (
        clusters is not None and pvals is not None)
    if add_highlight:
        if df_clst is not None:
            this_clst = df_clst.query(f'neuron == "{cell_name}"')
            pvals = this_clst.pval.values
            clusters = [_create_mask_from_window_str(twin, this_frate)
                        for twin in this_clst.window.values]
        add_highlights(this_frate, clusters, pvals, ax=ax[0],
                       p_threshold=p_threshold, min_pval=min_pval)

    plot_raster(spk.copy().pick_cells(cell_name), pick=0, groupby=groupby,
                colors=colors, ax=ax[1], eventplot_kwargs=eventplot_kwargs)
    ylim = ax[1].get_xlim()
    ax[0].set_xlim(ylim)

    ax[0].set_xlabel('')
    ax[0].set_ylabel('Spike rate (Hz)', fontsize=fontsize)
    ax[1].set_xlabel('Time (s)', fontsize=fontsize)
    ax[1].set_ylabel('Trials', fontsize=fontsize)

    return fig


def _create_mask_from_window_str(window, frate):
    twin = [float(x) for x in window.split(' - ')]
    mask = (frate.time.values >= twin[0]) & (frate.time.values <= twin[1])
    return mask


# consider moving to sarna?
# TODO: could infer x coords from plot (if lines are already present)
def add_highlights(arr, clusters, pvals, p_threshold=0.05, ax=None,
                   min_pval=0.001, bottom_extend=True, pval_text=True,
                   text_props=None):
    '''Highlight significant clusters along the last array dimension.

    Parameters
    ----------
    arr : xarray.DataArray | numpy.ndarray
        xarray Data array or numpy array. Used only for axis coordinates.
        If xarray, the last dimension is taken as x axis coordinates.
        If numpy array, the values are assumed to be the x axis coordinates.
    clusters : list of np.array
        List of boolean arrays, where each array contains cluster membership
        information (which points along the last array dimension contribute
        to the given cluster).
    pvals : list-like
        List or array of cluster p values.
    p_threshold : float
        Alpha significance threshold. Clusters with p value below this
        threshold will be shown.
    ax : matplotlib.Axes | None
        Axis to plot to. Optional, defaults to ``None`` - which creates the
        axis automatically.
    min_pval : float
        Minimum meaningful p value. P values below this value will be displayed
        as this value. This is to avoid p = 0, which can result in the cluster
        based permutation test, when no cluster from the null distribution had
        stronger summary statistic than the observed cluster. ``min_pval`` is
        best defined as ``1 / n_permutations``.
    bottom_extend : bool
        Whether to extend the lower limits of y axis when adding bottom
        significance bars. Defaults to ``True``.
    pval_text : bool
        Whether to add p value text boxes to respective cluster ranges in the
        plot. Defaults to ``True``.
    text_props : dict | None
        Dictionary with text properties for p value text boxes. If None,
        defaults to
        ``{'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.75, edgecolor='gray'}``.

    Returns
    -------
    ax : matplotlib.Axes
        Axis with the plot.
    '''
    import matplotlib.pyplot as plt
    from borsar.viz import highlight

    try:
        import xarray as xr
        has_xarray = True
    except ModuleNotFoundError:
        has_xarray = False

    if text_props is None and pval_text:
        text_props = dict(boxstyle='round', facecolor='white', alpha=0.75,
                          edgecolor='gray')

    if ax is None:
        ax = plt.gca()

    if pvals is None:
        return

    pvals_significant = pvals < p_threshold

    if isinstance(arr, np.ndarray):
        x_coords = arr
    elif has_xarray and isinstance(arr, xr.DataArray):
        last_dim = arr.dims[-1]
        x_coords = arr.coords[last_dim].values
    else:
        raise RuntimeError('The `arr` has to be either a numpy array or'
                           'xarray.DataArray.')

    extend_textbox_x = 4

    clusters_x_sorting = np.argsort([np.where(x)[0][0] for x in clusters])

    if pvals_significant.any():
        from borsar.viz import highlight
        from borsar.stats import format_pvalue

        ylm = ax.get_ylim()
        y_rng = np.diff(ylm)[0]

        # TODO: better estimate text y pos from textbox?
        if pval_text:
            y_step = 0.075 * y_rng
            text_y = ylm[1] - y_step
            ax.set_ylim([ylm[0], ylm[1] + 2 * y_step])

        sig_idx = np.where(pvals_significant)[0]
        sig_clusters = [clusters[ix] for ix in sig_idx]

        highlight(
            x_coords, sig_clusters, ax=ax,
            bottom_bar=True, bottom_extend=bottom_extend
        )

        # FIXME: put this into a separate function
        if pval_text:
            texts = list()
            for ix in clusters_x_sorting:
                if not pvals_significant[ix]:
                    continue

                this_pval = pvals[ix]
                text_x = np.mean(x_coords[clusters[ix]])

                if this_pval < min_pval:
                    p_txt = 'p < {:.3f}'.format(min_pval)
                else:
                    p_txt = format_pvalue(this_pval)

                this_text = ax.text(
                    text_x, text_y, p_txt,
                    bbox=text_props, horizontalalignment='center'
                )
                try:
                    textbox = this_text.get_window_extent()
                except RuntimeError:
                    ax.figure.canvas.draw()
                    textbox = this_text.get_window_extent()

                textbox.set_points(
                    np.array([
                        [textbox.x0 - extend_textbox_x, textbox.y0],
                        [textbox.x1 + extend_textbox_x, textbox.y1]
                    ])
                )

                if len(texts) > 0:
                    while textbox.count_overlaps(texts):
                        x, y = this_text.get_position()
                        this_text.set_position((x, y - y_step))
                        textbox = this_text.get_window_extent()
                texts.append(textbox)

    return ax


# - [ ] combine with layout functions from sarna
def align_axes_limits(axes=None, ylim=True, xlim=False):
    '''Align the limits of all ``axes``.'''
    import matplotlib.pyplot as plt

    if axes is None:
        axes = plt.gcf().get_axes()

    do_lim = list()
    if xlim: do_lim.append('x')
    if ylim: do_lim.append('y')

    limits = dict(x=[np.inf, -np.inf], y=[np.inf, -np.inf])
    iter = (axes if isinstance(axes, (list, np.ndarray))
            else axes.values() if isinstance(axes, dict) else None)

    for ax in iter:
        get_lim = dict(x=ax.get_xlim, y=ax.get_ylim)
        for lim in do_lim:
            this_lim = get_lim[lim]()
            if limits[lim][0] > this_lim[0]:
                limits[lim][0] = this_lim[0]
            if limits[lim][1] < this_lim[1]:
                limits[lim][1] = this_lim[1]

    iter = (axes if isinstance(axes, (list, np.ndarray))
            else axes.values() if isinstance(axes, dict) else None)
    for ax in iter:
        set_lim = dict(x=ax.set_xlim, y=ax.set_ylim)
        for lim in do_lim:
            set_lim[lim](limits[lim])


# TODO - move this to separate submodule .waveform ?
# TODO - the API could be better / simplified: low level function that
#        takes just waveform array and high-level function that takes
#        Spikes object and cell indices (currently only one cell is supported)
def calculate_perceptual_waveform_density(spk, cell_idx, take_n=15):
    """
    Experimental measure of "perceptual" waveform density.

    The goal is to provide a single value that would reflect the density of
    the waveform in the similar way we perceive it on waveform density plots.

    The measure is calculated as follows:
    - calculate 2d histogram of the waveform
    - correct the y range to remove extreme values that bias the 2d histogram
      calculation
    - take the top N values from each row
    - normalize these values by the maximum value
    - average the normalized values

    Parameters
    ----------
    spk : Spikes | SpikeEpochs
        Spike object to use.
    cell_idx : int
        Index of the cell to calculate the perceptual waveform density for.
    take_n : int
        How many top values to take from each row of the 2d histogram.

    Returns
    -------
    dns : float
        Perceptual waveform density.
    """
    # get waveform 2d histogram image
    hist, _, ybins = (
        _calculate_waveform_density_image(
            spk.waveform[cell_idx], 100)
    )

    # correct y range
    # TODO: would be cheaper to calculate the range in the function above
    sm = hist.sum(axis=0)
    msk = sm < 1e-6
    n_smp = len(msk)

    start_ix, end_ix = 0, n_smp

    if msk[0]:
        # trim from front
        for ix in range(n_smp):
            if not msk[ix]:
                break
        start_ix = ix

    if msk[-1]:
        # trim from back
        for ix in range(-1, -n_smp, -1):
            if not msk[ix]:
                break
        end_ix = ix

    # create the 2d hist image with corrected y range
    y_range = [ybins[start_ix], ybins[end_ix]]
    hist, _, ybins = (
        _calculate_waveform_density_image(
            spk.waveform[cell_idx], 100, y_range=y_range)
    )

    # calculate dns
    # TODO: a better way would be to calculate mean from a spatial window
    #       around max, not the top N values
    hist_sort = np.sort(hist)[:, ::-1]
    vals = (hist_sort[:, :take_n] / hist_sort.max()).mean(axis=-1)
    dns = vals.mean()

    return dns


def auto_multipanel(n_to_show, ax=None, figsize=None):
    '''Create a multipanel figure that fits at least ``n_to_show`` axes.'''
    import matplotlib.pyplot as plt

    n = np.sqrt(n_to_show)
    n_left = 0

    if ax is None:
        n *= 1.25
        n_cols = int(np.round(n))
        n_rows = int(np.ceil(n_to_show / n))

        if n_to_show > 1:
            n_left = (n_cols * n_rows) - n_to_show

            # check if one less row works better
            n_cols_try, n_rows_try = n_cols + 1, n_rows - 1
            n_left_try1 = (n_cols_try * n_rows_try) - n_to_show
            try1_good = n_left_try1 > 0 and n_left_try1 < n_left
            n_left_try1 += (1 - try1_good) * 100

            # also check if one less column is better
            n_cols_try2, n_rows_try2 = n_cols - 1, n_rows + 1
            n_left_try2 = (n_cols_try2 * n_rows_try2) - n_to_show
            try2_good = n_left_try2 > 0 and n_left_try2 < n_left
            n_left_try2 += (1 - try2_good) * 100

            if try1_good or try2_good:
                if n_left_try2 < n_left_try1:
                    n_cols, n_rows = n_cols_try2, n_rows_try2
                else:
                    n_cols, n_rows = n_cols_try, n_rows_try

            n_left = (n_cols * n_rows) - n_to_show
            if figsize is None:
                if n_cols == 1 and n_rows == 1:
                    figsize = None
                elif n_rows > 2 or n_cols > 2:
                    # some calculation
                    figsize = (n_cols * 1.35 * 1.5, n_rows * 1.5)

        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize,
                               constrained_layout=True)

        if n_left > 0:
            for this_ax in ax.ravel()[-n_left:]:
                this_ax.axis('off')

    return ax


def _simplify_axes(ax):
    if isinstance(ax, list):
        axes = np.array(ax)
    elif isinstance(ax, np.ndarray):
        axes = ax.ravel()
    else:
        axes = np.array([ax])
    return axes


def plot_isi(spk, picks=None, unit='ms', bins=None, min_spikes=20,
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
        Maximum ISI time to plot. If ``None`` the maximum ISI is set to 0.1 for
        ``unit == 's'`` and 100 for ``unit == 'ms'``.
    ax : matplotlib.Axes | None
        Axis to plot to. If ``None`` a new figure is created.

    Returns
    -------
    ax : matplotlib.Axes
        Axes with the plot.
    '''
    from .utils import _deal_with_picks
    from .spikes import Spikes

    msg = 'Currently only Spikes are supported in ``plot_isi()``.'
    assert(isinstance(spk, Spikes)), msg

    assert unit in ['s', 'ms']
    picks = _deal_with_picks(spk, picks)
    n_picks = len(picks)
    ax = auto_multipanel(n_picks, ax=ax)

    div = 1000 if unit == 'ms' else 1
    if max_isi is None:
        max_isi = 100 if unit == 'ms' else 0.1

    axes = _simplify_axes(ax)
    n_axes = len(axes)
    for idx, unit_idx in enumerate(picks):
        stamps = spk.timestamps[unit_idx]

        if len(stamps) > min_spikes:
            isi = np.diff(stamps / (spk.sfreq / div))
            isi = isi[isi < max_isi]
            n_isi = len(isi)
            use_bins = (bins if bins is not None
                        else max(min(250, int(n_isi / 100)), 10))
            axes[idx].hist(isi, bins=use_bins)
            axes[idx].set_ylabel('Count', fontsize=12)
            axes[idx].set_xlabel(f'ISI ({unit})', fontsize=12)
        axes[idx].set_title(spk.cell_names[unit_idx], fontsize=12)

    if n_axes > n_picks:
        for ix in range(n_picks, n_axes):
            axes[ix].set_xticks([])
            axes[ix].set_yticks([])

    return ax
