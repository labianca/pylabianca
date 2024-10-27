import numpy as np


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
                msg = (f'Removing {n_reject} bad waveforms for cell '
                       f'{spk.cell_names[cell_idx]}.')
                print(msg)

                spk.waveform[cell_idx] = np.delete(
                    spk.waveform[cell_idx], reject_idx, axis=0)
                spk.timestamps[cell_idx] = np.delete(
                    spk.timestamps[cell_idx], reject_idx)
