import warnings

import numpy as np
import pytest
import xarray as xr

import pylabianca as pln
from pylabianca.testing import random_xarray


def _condition_mean_difference(X, y):
    levels = np.unique(y)
    assert len(levels) == 2
    axis = (0, 1) if X.ndim == 3 else None
    return X[y == levels[1]].mean(axis=axis) - X[y == levels[0]].mean(
        axis=axis)


def _simple_decoding_score(X, y, time=None):
    """Small deterministic decoding stand-in for wrapper tests."""
    assert X.ndim == 3
    score = _condition_mean_difference(X, y)
    return xr.DataArray(score, dims=['time'], coords={'time': time})


def _fold_score(arr, target='cond'):
    values = arr.transpose('trial', 'cell', 'time').values
    y = arr.coords[target].values
    score = _condition_mean_difference(values, y)
    return xr.DataArray(
        np.stack([score, score + 0.1], axis=0),
        dims=['fold', 'time'],
        coords={'fold': [0, 1], 'time': arr.time.values},
        name='score'
    )


def _add_condition_signal(arr, cells, time_idx, signal=2.):
    levels = np.unique(arr.cond.values)
    trials = arr.cond.values == levels[-1]
    arr.data[np.ix_(cells, trials, time_idx)] += signal


def test_random_xarray_condition_signal_uses_cond_coord():
    arr = random_xarray(
        n_cells=4, n_trials=12, n_times=3,
        trial_condition_levels=(0, 1), signal=1., random_state=10)

    assert 'cond' in arr.coords
    np.testing.assert_array_equal(np.unique(arr.cond.values), [0, 1])
    assert int((arr.cond == 0).sum()) == int((arr.cond == 1).sum())

    effect = arr.sel(trial=arr.cond == 1).mean('trial')
    effect -= arr.sel(trial=arr.cond == 0).mean('trial')

    assert bool((effect > 0.5).all())


def test_frate_to_sklearn_selection_and_decimation():
    arr = random_xarray(
        n_cells=5, n_trials=12, n_times=8,
        trial_condition_levels=(0, 1), signal=1., random_state=0)
    block = np.tile([1, 2], 6)
    arr = arr.assign_coords(block=('trial', block))
    arr = arr.transpose('time', 'cell', 'trial')

    cell_names = arr.cell.values[[1, 3]]
    X, y, time = pln.decoding.frate_to_sklearn(
        arr, target='cond', select='block == 1',
        cell_names=cell_names, time_idx=[1, 3, 5], decim=2
    )

    expected = arr.transpose('trial', 'cell', 'time')
    expected = expected.sel(cell=cell_names).query({'trial': 'block == 1'})
    expected = expected.isel(time=[1, 3, 5])

    assert X.shape == (6, 2, 2)
    np.testing.assert_array_equal(X, expected.values[..., ::2])
    np.testing.assert_array_equal(y, expected.cond.values)
    np.testing.assert_array_equal(time, expected.time.values[::2])


def test_frate_to_sklearn_without_time_dimension():
    arr = random_xarray(
        n_cells=4, n_trials=10, n_times=1,
        trial_condition_levels=(0, 1), signal=1., random_state=1)
    arr = arr.isel(time=0)

    X, y, time = pln.decoding.frate_to_sklearn(arr, target='cond')

    assert X.shape == (10, 4)
    assert time is None
    np.testing.assert_array_equal(y, arr.cond.values)


def test_frates_dict_to_sklearn_uses_subject_cell_selection():
    frates = {
        'sub-01': random_xarray(
            n_cells=4, n_trials=10, n_times=6,
            trial_condition_levels=(0, 1), signal=1., random_state=2),
        'sub-02': random_xarray(
            n_cells=5, n_trials=10, n_times=6,
            trial_condition_levels=(0, 1), signal=1., random_state=3),
    }
    cell_names = {
        'sub-01': frates['sub-01'].cell.values[:2],
        'sub-02': frates['sub-02'].cell.values[[1, 3, 4]],
    }

    Xs, ys, time = pln.decoding.frates_dict_to_sklearn(
        frates, target='cond', cell_names=cell_names, decim=2
    )

    assert [X.shape for X in Xs] == [(10, 2, 3), (10, 3, 3)]
    np.testing.assert_array_equal(ys[0], frates['sub-01'].cond.values)
    np.testing.assert_array_equal(ys[1], frates['sub-02'].cond.values)
    np.testing.assert_array_equal(time, frates['sub-02'].time.values[::2])


def test_join_subjects_aligns_conditions_without_shuffle():
    y1 = np.array([1, 0, 1, 0])
    y2 = np.array([0, 1, 0, 1])
    X1 = np.arange(4 * 2 * 1).reshape(4, 2, 1)
    X2 = np.arange(100, 100 + 4 * 3 * 1).reshape(4, 3, 1)

    X, y = pln.decoding.join_subjects([X1, X2], [y1, y2], shuffle=False)

    np.testing.assert_array_equal(y, [0, 0, 1, 1])
    np.testing.assert_array_equal(X[:, :2], X1[[1, 3, 0, 2]])
    np.testing.assert_array_equal(X[:, 2:], X2[[0, 2, 1, 3]])


def test_join_subjects_shuffle_is_reproducible():
    y = np.array([0, 0, 1, 1, 0, 1])
    Xs = [
        np.arange(6 * 2 * 1).reshape(6, 2, 1),
        np.arange(100, 100 + 6 * 3 * 1).reshape(6, 3, 1),
    ]

    X1, y1 = pln.decoding.join_subjects(Xs, [y, y], random_state=7)
    X2, y2 = pln.decoding.join_subjects(Xs, [y, y], random_state=7)

    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_resample_decoding_with_arrays_returns_xarray():
    args = dict(
        n_trials=12, n_times=5, trial_condition_levels=(0, 1), signal=1.)
    Xs = [
        random_xarray(n_cells=3, random_state=4, **args).values.transpose(
            1, 0, 2),
        random_xarray(n_cells=2, random_state=5, **args).values.transpose(
            1, 0, 2),
    ]
    ys = [np.resize([0, 1], 12), np.resize([1, 0], 12)]
    time = np.linspace(-0.1, 0.3, 5)

    out = pln.decoding.resample_decoding(
        _simple_decoding_score, Xs=Xs, ys=ys, time=time, n_resamples=3
    )

    assert out.dims == ('resample', 'time')
    assert out.shape == (3, 5)
    np.testing.assert_array_equal(out.time.values, time)


def test_run_decoding_array_uses_condition_signal():
    signal = random_xarray(
        n_cells=4, n_trials=60, n_times=3,
        trial_condition_levels=(0, 1), signal=2., random_state=11)
    noise = random_xarray(
        n_cells=4, n_trials=60, n_times=3,
        trial_condition_levels=(0, 1), signal=1e-12, random_state=12)

    signal_X, signal_y, _ = pln.decoding.frate_to_sklearn(
        signal, target='cond')
    noise_X, noise_y, _ = pln.decoding.frate_to_sklearn(noise, target='cond')

    signal_score = pln.decoding.run_decoding_array(
        signal_X[..., 0], signal_y, n_splits=4, random_state=0).mean()
    noise_score = pln.decoding.run_decoding_array(
        noise_X[..., 0], noise_y, n_splits=4, random_state=0).mean()

    assert signal_score > 0.95
    assert noise_score < 0.75


def test_run_decoding_array_decodes_signal_timepoints():
    arr = random_xarray(
        n_cells=6, n_trials=80, n_times=6,
        trial_condition_levels=(0, 1), random_state=13)
    _add_condition_signal(arr, np.arange(6), np.array([2, 3]), signal=3.)

    X, y, time = pln.decoding.frate_to_sklearn(
        arr, target='cond', decim=1)
    scores = pln.decoding.run_decoding_array(
        X, y, n_splits=4, random_state=0, time=time)
    mean_scores = scores.mean('fold')

    assert scores.dims == ('fold', 'time')
    np.testing.assert_array_equal(scores.time.values, arr.time.values)
    assert bool((mean_scores.isel(time=[2, 3]) > 0.95).all())
    assert bool((mean_scores.isel(time=[0, 1, 4, 5]) < 0.75).all())


def test_run_decoding_array_time_generalization_tracks_pattern_change():
    arr = random_xarray(
        n_cells=4, n_trials=80, n_times=4,
        trial_condition_levels=(0, 1), random_state=14)
    _add_condition_signal(arr, np.array([0]), np.array([0, 1]), signal=3.)
    _add_condition_signal(arr, np.array([1]), np.array([2, 3]), signal=3.)

    X, y, time = pln.decoding.frate_to_sklearn(
        arr, target='cond', decim=1)
    scores = pln.decoding.run_decoding_array(
        X, y, n_splits=4, random_state=0, time=time,
        time_generalization=True)
    mean_scores = scores.mean('fold')

    assert scores.dims == ('fold', 'train_time', 'test_time')
    np.testing.assert_array_equal(scores.train_time.values, arr.time.values)
    np.testing.assert_array_equal(scores.test_time.values, arr.time.values)

    early = [0, 1]
    late = [2, 3]
    assert bool((mean_scores.isel(train_time=early, test_time=early)
                 > 0.95).all())
    assert bool((mean_scores.isel(train_time=late, test_time=late)
                 > 0.95).all())
    assert bool((mean_scores.isel(train_time=early, test_time=late)
                 < 0.75).all())
    assert bool((mean_scores.isel(train_time=late, test_time=early)
                 < 0.75).all())


def test_resample_decoding_validates_inputs():
    with pytest.raises(ValueError, match='Either frates or Xs and ys'):
        pln.decoding.resample_decoding(_simple_decoding_score)

    frates = {
        'sub-01': random_xarray(
            n_cells=3, n_trials=12, n_times=5,
            trial_condition_levels=(0, 1), signal=1., random_state=6)
    }
    with pytest.raises(ValueError, match='target must be provided'):
        pln.decoding.resample_decoding(_simple_decoding_score, frates=frates)


def test_decoding_input_errors_are_informative():
    from sklearn.svm import SVC

    arr = random_xarray(
        n_cells=3, n_trials=12, n_times=2,
        trial_condition_levels=(0, 1), signal=1., random_state=15)

    with pytest.raises(ValueError, match='specify target'):
        pln.decoding.frate_to_sklearn(arr)

    with pytest.raises(ValueError, match='specify target'):
        pln.decoding.frates_dict_to_sklearn({'sub-01': arr})

    with pytest.raises(ValueError, match='Cannot use PCA'):
        pln.decoding.run_decoding_array(
            np.zeros((12, 3)), arr.cond.values, clf=SVC(), n_pca=1)

    with pytest.raises(AssertionError):
        pln.decoding.run_decoding(arr, target='cond', decode_across='freq')


def test_max_corr_classifier_suppresses_constant_class_warning():
    clf = pln.decoding.maxCorrClassifier()
    X = np.array([
        [1., 1.],
        [1., 1.],
        [3., 0.],
        [3., 0.],
    ])
    y = np.array(['flat', 'flat', 'spread', 'spread'])
    clf.fit(X, y)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        pred = clf.predict(np.array([[1., 1.], [3., 0.]]))

    np.testing.assert_array_equal(pred, ['flat', 'spread'])
    assert not caught


def test_permute_decoding_average_folds_returns_permutation_xarray():
    arr = random_xarray(
        n_cells=3, n_trials=12, n_times=4,
        trial_condition_levels=(0, 1), signal=1., random_state=8)

    out = pln.decoding.permute(
        arr, _fold_score, target='cond', n_permutations=4,
        arguments={'target': 'cond'}
    )

    assert out.dims == ('permutation', 'time')
    assert out.shape == (4, 4)
    np.testing.assert_array_equal(out.time.values, arr.time.values)


def test_run_decoding_returns_time_xarray(monkeypatch):
    arr = random_xarray(
        n_cells=3, n_trials=12, n_times=4,
        trial_condition_levels=(0, 1), signal=1., random_state=9)

    def fake_run_decoding_array(
        X, y, n_splits=6, C=1., scoring='accuracy', n_jobs=1,
        time_generalization=False, random_state=None, clf=None, n_pca=0,
        time=None
    ):
        assert X.shape == (12, 3, 2)
        np.testing.assert_array_equal(y, arr.cond.values)
        np.testing.assert_array_equal(time, arr.time.values[::2])
        scores = np.zeros((n_splits, len(time)))
        return pln.decoding._scores_as_xarray(
            scores, scoring, n_splits, 'time', time, time_generalization)

    monkeypatch.setattr(
        pln.decoding, 'run_decoding_array', fake_run_decoding_array)

    scores = pln.decoding.run_decoding(
        arr, target='cond', decim=2, n_splits=2, random_state=0
    )

    assert scores.dims == ('fold', 'time')
    assert scores.shape == (2, 2)
    assert scores.name == 'accuracy'
    np.testing.assert_array_equal(scores.time.values, arr.time.values[::2])


def test_max_corr_classifier_predicts_nearest_class_average():
    clf = pln.decoding.maxCorrClassifier()
    X = np.array([
        [0.0, 1.0],
        [0.1, 0.9],
        [1.0, 0.0],
        [0.9, 0.1],
    ])
    y = np.array(['low', 'low', 'high', 'high'])

    clf.fit(X, y)
    pred = clf.predict(np.array([[0.2, 0.8], [0.8, 0.2]]))

    np.testing.assert_array_equal(pred, ['low', 'high'])
    assert clf.score(X, y) == 1.0


def test_select_n_best_cells_uses_largest_boolean_split():
    rng = np.random.default_rng(0)
    X = rng.normal(scale=0.1, size=(8, 4, 3))
    y = np.array([False, False, False, False, True, True, True, True])
    X[y, 2, :] += 5.0
    X[y, 0, :] += 1.0

    selected = pln.decoding.select_n_best_cells(X, y, select_n=1)

    np.testing.assert_array_equal(selected, [False, False, True, False])
