import warnings
import numpy as np

# avoid errors when no sklearn
try:
    from sklearn.base import BaseEstimator
except ImportError:
    BaseEstimator = object


def run_decoding_array(X, y, n_splits=6, C=1., scoring='accuracy',
                       n_jobs=1, time_generalization=False, random_state=None,
                       clf=None, n_pca=0, time=None):
    '''Perform decoding analysis.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features, n_times)
        The training input samples. Commonly the samples dimension consists of
        trials or subjects while the features dimension consists of cells or
        electrodes.
    y : array-like, shape (n_samples,)
        The target values (class labels).
    n_splits : int | str
        Number of cross-validation splits. If ``'loo'``, leave-one-out
        cross-validation is used.
    C : float
        Inverse of regularization strength.
    scoring : str
        Scoring metric.
    n_jobs : int
        Number of jobs to run in parallel.
    time_generalization : bool
        Whether to perform time generalization (training and testing also
        on different time points).
    random_state : int or None
        Random state for cross-validation.
    clf : None or sklearn classifier / pipeline
        If None, a linear SVM classifier with standard scaling is used.
    n_pca : int
        Number of principal components to use for dimensionality reduction. If
        0 (default), no dimensionality reduction is performed.

    Returns
    -------
    scores : array, shape (n_splits, n_times, n_times)
        Decoding scores.
    '''
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, LeaveOneOut
    from mne.decoding import GeneralizingEstimator, SlidingEstimator

    if n_pca > 0:
        if clf is not None:
            raise ValueError('Cannot use PCA and a custom classifier.'
                             ' You would have to construct you own pipeline.')

        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_pca)

    # handle data with only one time point / aggregated time window
    one_time_sample = False
    if X.ndim == 2:
        one_time_sample = True

    # k-fold object
    if isinstance(n_splits, str) and n_splits == 'loo':
        # use leave one out cross validation
        spl = LeaveOneOut()
    else:
        spl = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

    # classification pipeline
    if clf is None:
        steps = [StandardScaler(), SVC(C=C, kernel='linear')]
        if n_pca > 0:
            steps.insert(1, pca)
        clf = make_pipeline(*steps)

    # use simple sliding estimator or generalization across time
    if not one_time_sample:
        estimator = (SlidingEstimator if not time_generalization
                     else GeneralizingEstimator)
        estimator = estimator(
            clf, scoring=scoring,
            n_jobs=n_jobs, verbose=False
        )
    else:
        estimator = clf

    # do the k-fold
    scores = list()
    for train_index, test_index in spl.split(X, y):
        estimator.fit(X=X[train_index],
                      y=y[train_index])
        score = estimator.score(X=X[test_index],
                                y=y[test_index])
        scores.append(score)

    scores = np.stack(scores, axis=0)

    if time is not None:
        scores = _scores_as_xarray(scores, scoring, n_splits, 'time', time,
                                   time_generalization)

    return scores


# TODO: decode_across is not actually used
# CONSIDER: decim=None by default, decim=1 as no decimation may be confusing
# CONSIDER: supporting ``select`` to select conditions (useful only when a
#           dictionary of xarrays is passed, so multiple subjects)
def run_decoding(arr, target, decode_across='time', decim=1, n_splits=6, C=1.,
                 scoring='accuracy', n_jobs=1, time_generalization=False,
                 random_state=None, clf=None, n_pca=0):
    '''Perform decoding analysis using xarray as input.

    Parameters
    ----------
    arr : xarray.DataArray
        The data to use in classification. Should contain a ``'trials'``
        dimension and a dimension consistent with ``decode_across``.
    target : str
        Name of the variable to use as target.
    decode_across : str
        Name of the dimension to perform decoding across. For each element of
        this dimension a separate decoding will be performed. The default is
        ``'time'`` - which slides the decoding classifier across the time
        dimension.
    decim : int
        Decimation for the ``decode_across`` dimension. Default's to ``1``,
        which does not perform decimation.
    n_splits : int | str
        Number of cross-validation splits. If ``'loo'``, leave-one-out
        cross-validation is used.
    C : float
        Inverse of regularization strength.
    scoring : str
        Scoring metric.
    n_jobs : int
        Number of jobs to run in parallel.
    time_generalization : bool
        Whether to perform time generalization (training and testing also
        on different time points).
    random_state : int or None
        Random state for cross-validation.
    clf : None or sklearn classifier / pipeline
        If None, a linear SVM classifier with standard scaling is used.
    n_pca : int
        Number of principal components to use for dimensionality reduction. If
        0 (default), no dimensionality reduction is performed.

    Returns
    -------
    scores : xarray
        Decoding scores. The first dimension should consist of folds (separate
        cross-validation folds).
    '''
    import xarray as xr
    assert isinstance(arr, xr.DataArray)
    assert decode_across in arr.dims

    orig_dims = list(arr.dims)

    X, y, time_dim = frate_to_sklearn(
        arr, target=target, select=None, decim=decim)
    scores = run_decoding_array(
        X, y, n_splits=n_splits, C=C, scoring=scoring, n_jobs=n_jobs,
        time_generalization=time_generalization, random_state=random_state,
        clf=clf, n_pca=n_pca, time=time_dim
    )

    return scores


def _scores_as_xarray(scores, scoring, n_splits, decode_across, time_dim,
                      time_generalization):
    import xarray as xr

    name = scoring
    coords = {'fold': np.arange(n_splits)}
    if time_generalization:
        dims = ['fold', 'train_' + decode_across, 'test_' + decode_across]
        coords[dims[1]] = time_dim
        coords[dims[2]] = time_dim
    else:
        dims = ['fold'] + [decode_across]
        coords[decode_across] = time_dim

    scores = xr.DataArray(
        scores, dims=dims, coords=coords, name=name,
    )

    return scores


# TODO: later may be useful to make it accept arrays with different dimension
#       names
def frate_to_sklearn(frate, target=None, select=None,
                     cell_names=None, time_idx=None, decim=10):
    '''Formats xarray.DataArray into sklearn X and y data arrays.

    Parameters
    ----------
    frate : xarray.DataArray
        The data to turn in sklearn X and y format. Should contain a
        ``'trials'``, ``'cell'`` and ``'time'`` dimension.
    target : str
        Name of the variable to use as target.
    select : str | None
        Condition query to subselect trials. If ``None`` no trial selection
        is performed (all trials are used).
    cell_names : list of str | None
        Cell names to select.
    time_idx : int | array of int | None
        Select specific time point or time points. If ``None``, no specific
        time points are picked (apart from decimation controlled by ``decim``).
    decim : int
        Decimation factor.

    Returns
    -------
    X : numpy.ndarray
        2D array of n_observations x n_features (x n_times) shape. The time
        dimension is added only when it is present in the input xarray.
    y : numpy.ndarray
        1D array of n_observations length with target categories to classify.
    time : numpy.ndarray
        1D array of time labels after decimation.
    '''
    if target is None:
        raise ValueError('You have to specify specify target.')

    has_time = 'time' in frate.dims
    if has_time:
        fr = frate.transpose('trial', 'cell', 'time')
    else:
        fr = frate.transpose('trial', 'cell')

    # if concat_cond is not None:
    #     fr2 = frates['maint1'][subj].transpose(
    #         'trial', 'cell', 'time').sel(time=slice(-0.25, 2.5))

    if cell_names is not None:
        fr = fr.sel(cell=cell_names)

    if select is not None:
        fr = fr.query({'trial': select})

    if time_idx is not None:
        fr = fr.isel(time=time_idx)
        has_time = 'time' in fr.dims

    if has_time:
        time = fr.time.values[::decim]
        X = fr.values[..., ::decim]
    else:
        time = None
        X = fr.values

    y = fr.coords[target].values

    return X, y, time


def frates_dict_to_sklearn(frates, target=None, select=None,
                           cell_names=None, time_idx=None, decim=10):
    '''Get all subjects from frates dictionary.

    The ``frates`` is a dictionary of subject -> firing rate xarray mappings.

    Parameters
    ----------
    frates : dict
        Dictionary of the form {subject_string: firing rate xarray}.
    target : str
        Name of the variable to use as target.
    select : str, optional
        Query string to select trials. Default is ``None``, which does not
        subselect trials (all trials are used).
    cell_names : dict of list of str, optional
        Dictionary with subject id as keys and values being list of cell names
        to use. Defaults to ``None``, which uses all cells.
    time_idx : int, optional
        Time index or time range to use (as time indices). Defaults to
        ``None``, which uses all time points.
    decim : int
        Decimation factor.

    Returns
    -------
    Xs : list of arrays
        List of (n_trials, n_cells, n_time) firing rate arrays extracted from
        the ``frates`` dictionary. Each list element represents one subject.
    ys : list of arrays
        List of (n_trials,) target values extracted from the ``frates``
        dictionary. Each list element represents one subject.
    full_time : np.array | None
        Full time vector (in seconds). ``None`` if no time dimension was
        present in the the xarrays in ``frates`` dictionary.
    '''
    Xs = list()
    ys = list()

    if target is None:
        raise ValueError('You have to specify specify target.')

    if cell_names is None:
        subjects = frates.keys()
    else:
        subjects = cell_names.keys()

    for subj in subjects:
        this_cell_names = (cell_names[subj] if cell_names is not None
                           else cell_names)
        X, this_y, full_time = frate_to_sklearn(
            frates[subj], select=select, target=target,
            cell_names=this_cell_names, time_idx=time_idx, decim=decim
        )

        # add to the list
        Xs.append(X)
        ys.append(this_y)

    return Xs, ys, full_time


# TODO: DOC - explain shuffle argument better - it does not abolish the
#             relationship between trials of X and y, but only shuffles the
#             order of trials within each condition (y value)
def join_subjects(Xs, ys, random_state=None, shuffle=True):
    '''Concatenate subjects keeping target the same but shuffling trials
    within target categories.

    The concatenated array is trials x subjects_cells x time.

    Parameters
    ----------
    Xs : list of arrays
        List of (n_trials, n_cells, n_time) firing rate arrays.
    ys : list of arrays
        List of (n_trials,) target values.
    random_state : int, optional
        Random state to use for shuffling within-condition trials per subject
        before joining the arrays into one "pseudo-population".
    shuffle : bool, optional
        Whether to shuffle trials within conditions before joining the arrays.

    Returns
    -------
    X : array
        (n_trials, n_cells, n_time) firing rate array. The cells dimension
        is a pseudo-population pooled from all the subjects.
    y : array
        (n_trials,) target values.
    '''
    n_arrays = len(Xs)
    new_Xs, new_ys = list(), list()

    if random_state is not None:
        rnd = np.random.default_rng(random_state)
    else:
        rnd = None

    for idx in range(n_arrays):

        # shuffle
        X, y = Xs[idx], ys[idx]

        if shuffle:
            X, y = shuffle_trials(X, y, random_state=rnd)

            # sort trials by dependent value
            srt = np.argsort(y)
        else:
            # normal sorting can change the order of trials within conditions
            # so here, we do simpler reordering that retains this order
            categories = np.unique(y)
            srt = np.concatenate([np.where(y == cat)[0]
                                  for cat in categories])

        X = X[srt, :]
        y = y[srt]

        # add to the list
        new_Xs.append(X)
        new_ys.append(y)

    # make sure trials are aligned
    assert all([(y == new_ys[0]).all() for y in new_ys])

    X = np.concatenate(new_Xs, axis=1)
    y = new_ys[0]

    return X, y


# CONSIDER: to not have to copy arr, we can allow `target` to get a vector of
#           values
def permute(arr, decoding_fun, target=None, n_permutations=200, n_jobs=1,
            average_folds=True, arguments=dict()):
    import pandas as pd
    import xarray as xr

    scores = list()
    arr = arr.copy()  # we copy to modify target
    target_coord = arr.coords[target].values

    if 'target' not in arguments:
        arguments['target'] = target

    if n_jobs > 1:
        from joblib import Parallel, delayed

        # preprocess in parallel only at the top (permutation) level
        arguments['n_jobs'] = 1

        scores = Parallel(n_jobs=n_jobs)(
            delayed(_do_permute)(
                arr, decoding_fun, target_coord,
                average_folds=average_folds,
                arguments=arguments
            )
            for perm_idx in range(n_permutations)
        )
    else:
        for _ in range(n_permutations):
            score = _do_permute(
                arr, decoding_fun, target_coord, average_folds=average_folds,
                arguments=arguments)
            scores.append(score)

    # join the results
    if isinstance(scores[0], xr.DataArray):
        perm = pd.Index(np.arange(n_permutations), name='permutation')
        scores = xr.concat(scores, perm)
    else:
        scores = np.stack(scores, axis=0)
    return scores


def _do_permute(arr, decoding_fun, target_coord, average_folds=True,
                arguments=dict()):
    # permute target
    np.random.shuffle(target_coord)

    scr = decoding_fun(arr, **arguments)

    if average_folds:
        scr = scr.mean(dim='fold')

    return scr


def resample_decoding(decoding_fun, frates=None, target=None, Xs=None, ys=None,
                      time=None, arguments=dict(), n_resamples=20, n_jobs=1,
                      permute=False, select_trials=None, decim=None):
    """Resample a decoding analysis.

    The resampling is done by rearranging trials within each subject,
    matching the ``target`` category across subjects.

    Parameters
    ----------
    decoding_fun : callable
        Decoding function to use. Must accept ``X`` and ``y`` as first two
        arguments, and allow for ``time`` keyword argument. When given the
        ``time`` argument, the function must return a xarray DataArray with
        decoding scores.
    frates : dict | None
        Dictionary of the form {subject_string: firing rate xarray}. If
        ``None``, the ``Xs`` and ``ys`` arguments must be provided.
    target : str | None
        Target category to use for decoding. Has to be provided if ``frates``
        is used.
    Xs : list of arrays | None
        List of (n_trials, n_cells, n_time) firing rate arrays (one array per
        subject / session). If ``None``, the ``frates`` argument must be
        provided.
    ys : list of arrays | None
        List of (n_trials,) target values (one array per subject / session).
        If ``None``, the ``frates`` argument must be provided.
    time : array | None
        If data is provided as ``Xs`` and ``ys`` arrays, a time vector is
        needed.
    arguments : dict, optional
        Additional arguments to pass to the ``decoding_fun``.
    n_resamples : int, optional
        Number of resamples to perform. Defaults to 20.
    n_jobs : int, optional
        Number of jobs to use for resampling. Defaults to 1, higher values
        will use joblib to parallelize the resampling.
    permute : bool | array, optional
        Whether to permute the trials within each subject. If ``True``, the
        trials are permuted randomly. If an array, the array is used to
        permute the trials. Defaults to ``False``, which does not perform
        permutation.
    select_trials : str, optional
        A string query used to select trials out of subject-specific xarrays.
        Defaults to ``None``, which uses all trials.
    decim : int | None, optional
        Temporal decimation factor. Defaults to ``None``, which does not
        perform decimation.

    Returns
    -------
    score_resamples : xarray.DataArray | numpy.ndarray
        Resampled decoding scores. If the ``decoding_fun`` returns a xarray
        DataArray, the resampled scores are also returned as a DataArray.
    """
    import xarray as xr

    # check if correct arguments are provided
    if frates is None:
        if Xs is None or ys is None:
            raise ValueError('Either frates or Xs and ys must be provided.')
    else:
        if target is None:
            raise ValueError('target must be provided if frates is used.')

        Xs, ys, time = frates_dict_to_sklearn(
            frates, target=target, select=select_trials, decim=decim)

    if isinstance(permute, bool) and permute:
        ys = [np.random.permutation(y) for y in ys]

    # split into n_jobs = 1 or n_resamples = 1
    # and n_jobs > 1 (joblib)
    if n_jobs == 1 or n_resamples == 1:
        score_resamples = [
            _do_resample(Xs, ys, decoding_fun, arguments,
                         time=time)
            for resample_idx in range(n_resamples)
        ]
    else:
        from joblib import Parallel, delayed

        score_resamples = Parallel(n_jobs=n_jobs)(
            delayed(_do_resample)(
                Xs, ys, decoding_fun, arguments, time=time)
            for resample_idx in range(n_resamples)
        )

    # join the results
    if isinstance(score_resamples[0], xr.DataArray):
        import pandas as pd
        resamples = pd.Index(np.arange(n_resamples), name='resample')
        score_resamples = xr.concat(score_resamples, resamples)
    else:
        score_resamples = np.stack(score_resamples, axis=0)
    return score_resamples


def _do_resample(Xs, ys, decoding_fun, arguments, time=None):
    X, y = join_subjects(Xs, ys)

    # do the actual decoding
    return decoding_fun(X, y, time=time, **arguments)


## a variant of this will be needed later in join_sessions
def _count_trials(Xs):
    '''Check trials foe each array in the list.'''
    # check n trials (across subjects)

    n_tri = np.array([X.shape[0] for X in Xs])
    assert (n_tri[0] == n_tri).all()
    n_tri = n_tri[0]

    return n_tri


def shuffle_trials(*arrays, random_state=None):
    '''Perform the same trial shuffling for multiple arrays.'''
    n_arrays = len(arrays)
    array_len = [len(x) for x in arrays]
    if n_arrays > 1:
        assert all([array_len[0] == x for x in array_len[1:]])

    n_trials = array_len[0]
    tri_idx = np.arange(n_trials)
    if random_state is None:
        np.random.shuffle(tri_idx)
    else:
        random_state.shuffle(tri_idx)

    shuffled = tuple(arr[tri_idx] for arr in arrays)
    if n_arrays == 1:
        shuffled = shuffled[0]
    return shuffled


def select_n_best_cells(X, y, select_n=1):
    '''Select n best cells (used on training data).

    Parameters
    ----------
    X : numpy.ndarray
        sklearn X array: ``n_observations x n_features (x n_samples)``.
    y : numpy.ndarray
        Vector of class id to predict.
    select_n : int
        The number of best-performing units to select.
    '''
    from scipy.stats import ttest_ind

    t_val_per_cell, _ = np.abs(ttest_ind(X[y], X[~y]))
    if t_val_per_cell.ndim > 1:
        # max over time
        t_val_per_cell = t_val_per_cell.max(axis=1)
    tval_ord = t_val_per_cell.argsort()[::-1]

    # return selector array
    n_features = t_val_per_cell.shape[0]
    sel = np.zeros(n_features, dtype='bool')
    sel[tval_ord[:select_n]] = True
    return sel


def correlation(X1, X2):
    '''Correlate two arrays.'''
    ncols1 = X1.shape[1]
    rval = np.corrcoef(X1, X2, rowvar=False)
    rval_sel = rval[:ncols1, ncols1:]
    return rval_sel


# TODO: profile and consider using a better correlation function
#       (numba for example)
class maxCorrClassifier(BaseEstimator):
    '''Simple implementation of maxCorr classifier.'''
    def __init__(self):
        '''Create an instance of maxCorr classifier.'''
        pass

    def fit(self, X, y, scoring=None):
        '''Fit maxCorr to training data.

        Parameters
        ----------
        X : np.array
            Training data to use in classification. n_observations x n_features
            numpy array.
        y : np.array
            Target classes to classify. n_observations vector.
        scoring : str | None
            Scikit-learn scoring method.

        Returns
        -------
        self : pylabianca.decoding.maxCorr
            Fit classifier to training data. Works in-place so the output
            does not have to be stored (but it's useful for chaining).
        '''
        from sklearn.utils.validation import check_X_y
        from sklearn.utils.multiclass import unique_labels

        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.class_averages_ = list()
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        for cls in self.classes_:
            msk = y == cls
            avg = X[msk, :].mean(axis=0)
            self.class_averages_.append(avg)

        self.class_averages_ = np.stack(self.class_averages_, axis=1)
        self.scoring = 'accuracy' if scoring is None else scoring

        return self

    def predict(self, X):
        '''
        Predict classes.

        Parameters
        ----------
        X : numpy.array
            Test data to use in predicting classes. Should be an
            n_observations x n_features numpy array.

        Returns
        -------
        y_pred : numpy.array
            Vector of predicted classes.
        '''
        from sklearn.utils.validation import check_array, check_is_fitted

        # Check if fit has been called
        check_is_fitted(self)
        X = check_array(X)

        # check if any of the classes is constant
        has_constant_class = (
            self.class_averages_ == self.class_averages_[[0], :]
            ).all(axis=0).any()

        if not has_constant_class:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore',
                    message='invalid value encountered in true_divide',
                    category=RuntimeWarning
                )
                r = correlation(self.class_averages_, X.T)
        else:
            distance = self.class_averages_[..., None] - X.T[:, None, :]
            r = np.linalg.norm(distance, axis=0) * -1

        bad_trials = np.isnan(r).all(axis=0)
        if bad_trials.any():
            distance = (self.class_averages_[..., None]
                        - X.T[:, None, bad_trials])
            r[:, bad_trials] = np.linalg.norm(distance, axis=0) * -1

        # pick class with best correlation:
        r_best = r.argmax(axis=0)
        y_pred = self.classes_[r_best]

        return y_pred

    def score(self, X=None, Y=None):
        '''
        Score classification.

        The scoring provided at initialization is used.

        Parameters
        ----------
        X : numpy.array
            Test data to use in predicting classes. Should be an
            n_observations x n_features numpy array.
        y : numpy.array
            Correct class labels. n_observations vector.
        '''
        from sklearn.metrics import get_scorer

        scorer = get_scorer(self.scoring)
        score = scorer(self, X, Y)
        return score
