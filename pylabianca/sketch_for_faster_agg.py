from itertools import product
import numpy as np


def _aggregate_data(arr_dict, var_names, omit_nan=False):
    from numbagg import group_nanmean

    keys = list(arr_dict.keys())
    n_cells_total = sum(v.shape[0] for v in arr_dict.values())
    var_levels = get_var_levels(arr_dict, var_names)
    new_frs, _ = combine_vars_dict(arr_dict, var_names, var_levels)

    n_levels = [len(x) for x in var_levels]
    has_time = 'time' in arr_dict[keys[0]].dims
    n_times = len(arr_dict[keys[0]].time) if has_time else None

    # generate correct reshapes and transpositions
    shape1, shape2, shape3, transp1, transp2 = _gen_shapes_and_transpositions(
        n_cells_total, n_times, n_levels)

    # aggregate
    # ---------
    start_idx = 0
    frs_agg = np.zeros(shape1)
    for sub_ses in new_frs.keys():
        test_fr = new_frs[sub_ses]
        values = test_fr.values
        labels = test_fr['combined'].values
        n_cells = values.shape[0]
        frs_agg[start_idx:start_idx + n_cells] = group_nanmean(
            values, labels, axis=1)
        start_idx += n_cells

    if omit_nan:
        use_axis = (1, 2) if has_time else 1
        nan_msk = np.isnan(frs_agg).any(axis=use_axis)

        if nan_msk.any():
            print(f'Ommiting {nan_msk.sum()} NaN observations '
                  f'({nan_msk.mean() * 100:.1f} %)')
            frs_agg = frs_agg[~nan_msk]
            n_cells_total = frs_agg.shape[0]

            # FIXME: in this case it is done twice, could be done only once ...
            shape1, shape2, shape3, transp1, transp2 = (
                _gen_shapes_and_transpositions(
                    n_cells_total, n_times, n_levels))

    return (frs_agg, n_cells_total, has_time, shape2, shape3,
            transp1, transp2, n_levels)


def get_var_levels(arr_dct, var_names):
    # ENH: the var_levels could be init to correct dtype
    var_levels = list()
    for key_idx, key in enumerate(arr_dct.keys()):
        this_arr = arr_dct[key]
        for idx, var_name in enumerate(var_names):
            current_unique = np.unique(this_arr[var_name].values)
            if key_idx == 0:
                var_levels.append(current_unique)
            else:
                var_levels[idx] = np.unique(
                    np.append(var_levels[idx], current_unique)
                )
    return var_levels


def combine_vars_dict(arr_dict, var_names, var_levels):
    new_dict = dict()
    for key in arr_dict.keys():
        this_arr = arr_dict[key]
        new_vals, transl = combine_vars(this_arr, var_names, var_levels)
        this_arr = this_arr.assign_coords(combined=('trial', new_vals))
        new_dict[key] = this_arr

    return new_dict, transl


def combine_vars(arr, var_names, var_levels):
    vars = [arr[name].values for name in var_names]
    n_tri = len(vars[0])

    new_values = np.zeros(n_tri, dtype=int)
    translation = dict()
    msk = np.ones(n_tri, dtype=bool)
    for idx, combination in enumerate(product(*var_levels)):
        translation[idx] = combination
        this_msk = msk.copy()
        for var_idx, var in enumerate(vars):
            this_msk = this_msk & (var == combination[var_idx])
        new_values[this_msk] = idx

    return new_values, translation


def _gen_shapes_and_transpositions(n_cells_total, n_times, n_levels):
    has_time = n_times is not None
    n_times = [n_times] if has_time else list()
    n_factors = len(n_levels)

    # n_var1, n_var2 = n_levels
    level_comb = np.prod(n_levels)

    # shape1 - shape of numbagg aggregated data, condition combinations last
    #          as one dim, for example: cell x time x (factor1 * factor2)
    shape1 = tuple([n_cells_total] + n_times + [level_comb])

    # from aggregated data to permutation data (conditions as first dims):
    # * reshape to shape2 (unrolls conditions)
    # * then transpose to transp1 (factor1 x factor2 x cell x time)
    shape2 = tuple([n_cells_total] + n_times + n_levels)
    n_dim = len(shape2)
    last_transp = (1,) if has_time else ()
    transp1 = tuple(range(n_dim - n_factors, n_dim)) + (0,) + last_transp

    # back to ANOVA shape (n_cells x (factor1 * factor2) x time)
    # transp2, shape3
    last_transp = (n_dim - 1,) if has_time else ()
    transp2 = (n_factors,) + tuple(range(n_factors)) + last_transp
    shape3 = tuple([n_cells_total] + [level_comb] + n_times)

    return shape1, shape2, shape3, transp1, transp2

