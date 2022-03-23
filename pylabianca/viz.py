import numpy as np
import matplotlib.pyplot as plt


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
            msg = ('Xarray contains more than one cell to plot - this is '
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
# - [x] set y limits
# - [x] plot_waveform function
# - [ ] kind='line' ?
# - [ ] upsample in x dim?
def plot_waveform(spk, pick=0, upsample=False, ax=None):
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
        fig, ax = plt.subplots()

    ax.imshow(hist.T, alpha=alpha_sum, vmax=max_lim, origin='lower',
              extent=(time_edges[0], time_edges[-1], ybins[0], ybins[-1]),
              aspect='auto')
    ax.set_xlabel('Time (ms)', fontsize=14)
    ax.set_ylabel('Amplitude ($\mu$V)', fontsize=14)
