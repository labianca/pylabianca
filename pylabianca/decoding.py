import warnings
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels




# TODO: decimation should likely be done outside of this function
def run_decoding(X, y, decim=1, n_splits=6, C=1., scoring='accuracy',
                 n_jobs=4, time_generalization=False, random_state=None,
                 clf=None, n_pca=0, feature_selection=None):
    '''Perform decoding analysis with a linear SVM classifier.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features, n_times)
        The training input samples. Commonly the samples dimension consists of
        trials or subjects while the features dimension consists of cells or
        electrodes.
    y : array-like, shape (n_samples,)
        The target values (class labels).
    decim : int
        Decimation factor for the time dimension.
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
    feature_selection : function | None
        Function that takes ``X`` and ``y`` array from training set and
        returns a boolean array of shape (n_features,) that indicates which
        features to use for training. If None (default), no feature selection
        is performed.

    Returns
    -------
    scores : array, shape (n_splits, n_times, n_times)
        Decoding scores.
    sel_time : array, shape (n_times,)
        Selected time points. Can be ignored if ``decim==1``.
    '''
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, LeaveOneOut
    from mne.decoding import GeneralizingEstimator, SlidingEstimator

    if n_pca > 0:
        if clf is not None:
            raise ValueError('Cannot use PCA and a custom classifier.')

        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_pca)

    # handle data with only one time point / aggregated time window
    one_timesample = False
    if X.ndim == 2:
        X = X[:, :, np.newaxis]
    if X.shape[-1] == 1:
        decim = 1
        one_timesample = True

    # decimate original data
    sel_time = slice(0, X.shape[-1], decim)
    Xrs = X[..., sel_time]

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
    if not one_timesample:
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
    for train_index, test_index in spl.split(Xrs, y):
        if feature_selection is not None:
            sel = feature_selection(Xrs[train_index, :], y[train_index])
        else:
            sel = slice(None)

        estimator.fit(X=Xrs[train_index][:, sel],
                      y=y[train_index])
        score = estimator.score(X=Xrs[test_index][:, sel],
                                y=y[test_index])
        scores.append(score)

    scores = np.stack(scores, axis=0)
    return scores, sel_time


# TODO: returning ``full_time`` does not make much sense because
#       this function does not perform any decimation
def frate_to_sklearn(frate, target=None, select=None,
                     cell_names=None, time_idx=None):
    '''Format frates xarray into sklearn X, y data arrays.

    Can concatenate conditions if needed.
    '''
    if target is None:
        raise ValueError('You have to specify specify target.')

    if 'time' in frate.dims:
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

    full_time = fr.time.values if 'time' in fr.dims else None
    X = fr.values
    y = fr.coords[target].values

    return X, y, full_time


def frates_dict_to_sklearn(frates, target=None, select=None,
                           cell_names=None, time_idx=None):
    '''Get all subjects from frates dictionary.

    The ``frates`` is a dictionary of epoch types / conditions where each
    item is a dictionary of subject -> firing rate xarray mappings.

    Parameters
    ----------
    frates : dict
        Dictionary with epoch types / conditions as keys and
        {subject: firing rate xarray} dictionaries as keys.
    target : str
        Name of the variable to use as target.
    cond : str
        Epoch type / condition to choose.
    select : str, optional
        Query string to select trials. Default is ``None``, which does not
        subselect trials (all trials are used).
    cell_names : dict of list of str, optional
        Dictionary with subject id as keys and values being list of cell names
        to use. Defaults to ``None``, which uses all cells.
    time_idx : int, optional
        Time index or time range to use (as time indices). Defaults to
        ``None``, which uses all time points.

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
            cell_names=this_cell_names, time_idx=time_idx)

        # add to the list
        Xs.append(X)
        ys.append(this_y)

    return Xs, ys, full_time


def join_subjects(Xs, ys, random_state=None, shuffle=True):
    '''Concatenate subjects keeping target the same but shuffling tirals
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


def shuffle_trials(*arrays, random_state=None):
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
    ncols1 = X1.shape[1]
    rval = np.corrcoef(X1, X2, rowvar=False)
    rval_sel = rval[:ncols1, ncols1:]
    return rval_sel


# TODO: add option to correlate with single-trials, not only class-averages
class maxCorrClassifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
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

        return self

    def predict(self, X):
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