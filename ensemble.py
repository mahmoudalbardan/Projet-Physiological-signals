from __future__ import division
import numpy as np
from numpy import ma
from sklearn.tree.tree import BaseDecisionTree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.forest import ForestRegressor
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils import check_X_y

#from tree import DecisionTreeQuantileRegressor
#from tree import ExtraTreeQuantileRegressor
#from utils import weighted_percentile

def weighted_percentile(a, q, weights=None, sorter=None):
    """
    Returns the weighted percentile of a at q given weights.

    Parameters
    ----------
    a: array-like, shape=(n_samples,)
        samples at which the quantile.

    q: int
        quantile.

    weights: array-like, shape=(n_samples,)
        weights[i] is the weight given to point a[i] while computing the
        quantile. If weights[i] is zero, a[i] is simply ignored during the
        percentile computation.

    sorter: array-like, shape=(n_samples,)
        If provided, assume that a[sorter] is sorted.

    Returns
    -------
    percentile: float
        Weighted percentile of a at q.

    References
    ----------
    1. https://en.wikipedia.org/wiki/Percentile#The_Weighted_Percentile_method

    Notes
    -----
    Note that weighted_percentile(a, q) is not equivalent to
    np.percentile(a, q). This is because in np.percentile
    sorted(a)[i] is assumed to be at quantile 0.0, while here we assume
    sorted(a)[i] is given a weight of 1.0 / len(a), hence it is at the
    1.0 / len(a)th quantile.
    """
    if weights is None:
        weights = np.ones_like(a)
    if q > 100 or q < 0:
        raise ValueError("q should be in-between 0 and 100, "
                         "got %d" % q)

    a = np.asarray(a, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if len(a) != len(weights):
        raise ValueError("a and weights should have the same length.")

    if sorter is not None:
        a = a[sorter]
        weights = weights[sorter]

    nz = weights != 0
    a = a[nz]
    weights = weights[nz]

    if sorter is None:
        sorted_indices = np.argsort(a)
        sorted_a = a[sorted_indices]
        sorted_weights = weights[sorted_indices]
    else:
        sorted_a = a
        sorted_weights = weights

    # Step 1
    sorted_cum_weights = np.cumsum(sorted_weights)
    total = sorted_cum_weights[-1]

    # Step 2
    partial_sum = 100.0 / total * (sorted_cum_weights - sorted_weights / 2.0)
    start = np.searchsorted(partial_sum, q) - 1
    if start == len(sorted_cum_weights) - 1:
        return sorted_a[-1]
    if start == -1:
        return sorted_a[0]

    # Step 3.
    fraction = (q - partial_sum[start]) / (partial_sum[start + 1] - partial_sum[start])
    return sorted_a[start] + fraction * (sorted_a[start + 1] - sorted_a[start])



class BaseTreeQuantileRegressor(BaseDecisionTree):
    def predict(self, X, quantile=None, check_input=False):
        """
        Predict regression value for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        quantile : int, optional
            Value ranging from 0 to 100. By default, the mean is returned.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = [n_samples]
            If quantile is set to None, then return E(Y | X). Else return
            y such that F(Y=y | x) = quantile.
        """
        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        if quantile is None:
            return super(BaseTreeQuantileRegressor, self).predict(X, check_input=check_input)

        quantiles = np.zeros(X.shape[0])
        X_leaves = self.apply(X)
        unique_leaves = np.unique(X_leaves)
        for leaf in unique_leaves:
            quantiles[X_leaves == leaf] = weighted_percentile(
                self.y_train_[self.y_train_leaves_ == leaf], quantile)
        return quantiles

    def fit(self, X, y, sample_weight=None, check_input=True,
            X_idx_sorted=None):
        """
        Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.

        Returns
        -------
        self : object
            Returns self.
        """
        # y passed from a forest is 2-D. This is to silence the
        # annoying data-conversion warnings.
        y = np.asarray(y)
        if np.ndim(y) == 2 and y.shape[1] == 1:
            y = np.ravel(y)

        # apply method requires X to be of dtype np.float32
        X, y = check_X_y(
            X, y, accept_sparse="csc", dtype=np.float32, multi_output=False)
        super(BaseTreeQuantileRegressor, self).fit(
            X, y, sample_weight=sample_weight, check_input=check_input,
            X_idx_sorted=X_idx_sorted)
        self.y_train_ = y

        # Stores the leaf nodes that the samples lie in.
        self.y_train_leaves_ = self.tree_.apply(X)
        return self


class DecisionTreeQuantileRegressor(DecisionTreeRegressor, BaseTreeQuantileRegressor):
    """A decision tree regressor that provides quantile estimates.

    Parameters
    ----------
    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    splitter : string, optional (default="best")
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_features : int, float, string or None, optional (default=None)
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    presort : bool, optional (default=False)
        Whether to presort the data to speed up the finding of best splits in
        fitting. For the default settings of a decision tree on large
        datasets, setting this to true may slow down the training process.
        When using either a smaller dataset or a restricted depth, this may
        speed up the training.

    Attributes
    ----------
    feature_importances_ : array of shape = [n_features]
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

    max_features_ : int,
        The inferred value of max_features.

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree object
        The underlying Tree object.

    y_train_ : array-like
        Train target values.

    y_train_leaves_ : array-like.
        Cache the leaf nodes that each training sample falls into.
        y_train_leaves_[i] is the leaf that y_train[i] ends up at.
    """
    def __init__(self,
                 criterion="mse",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 presort=False):
        super(DecisionTreeQuantileRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state,
            presort=presort)


class ExtraTreeQuantileRegressor(ExtraTreeRegressor, BaseTreeQuantileRegressor):
    def __init__(self,
                 criterion='mse',
                 splitter='random',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 random_state=None,
                 max_leaf_nodes=None):
        super(ExtraTreeQuantileRegressor, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            random_state=random_state)



def generate_sample_indices(random_state, n_samples):
    """
    Generates bootstrap indices for each tree fit.

    Parameters
    ----------
    random_state: int, RandomState instance or None
        If int, random_state is the seed used by the random number generator.
        If RandomState instance, random_state is the random number generator.
        If None, the random number generator is the RandomState instance used
        by np.random.

    n_samples: int
        Number of samples to generate from each tree.

    Returns
    -------
    sample_indices: array-like, shape=(n_samples), dtype=np.int32
        Sample indices.
    """
    random_instance = check_random_state(random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    return sample_indices


class BaseForestQuantileRegressor(ForestRegressor):
    def fit(self, X, y):
        """
        Build a forest from the training set (X, y).

        Parameters
        ----------
        X : array-like or sparse matrix, shape = [n_samples, n_features]
            The training input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            The target values (class labels) as integers or strings.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        X_idx_sorted : array-like, shape = [n_samples, n_features], optional
            The indexes of the sorted training input samples. If many tree
            are grown on the same dataset, this allows the ordering to be
            cached between trees. If None, the data will be sorted here.
            Don't use this parameter unless you know what to do.

        Returns
        -------
        self : object
            Returns self.
        """
        # apply method requires X to be of dtype np.float32
        X, y = check_X_y(
            X, y, accept_sparse="csc", dtype=np.float32, multi_output=False)
        super(BaseForestQuantileRegressor, self).fit(X, y)

        self.y_train_ = y
        self.y_train_leaves_ = -np.ones((self.n_estimators, len(y)), dtype=np.int32)
        self.y_weights_ = np.zeros_like((self.y_train_leaves_), dtype=np.float32)

        for i, est in enumerate(self.estimators_):
            if self.bootstrap:
                bootstrap_indices = generate_sample_indices(
                    est.random_state, len(y))
            else:
                bootstrap_indices = np.arange(len(y))

            est_weights = np.bincount(bootstrap_indices, minlength=len(y))
            y_train_leaves = est.y_train_leaves_
            for curr_leaf in np.unique(y_train_leaves):
                y_ind = y_train_leaves == curr_leaf
                self.y_weights_[i, y_ind] = (
                    est_weights[y_ind] / np.sum(est_weights[y_ind]))

            self.y_train_leaves_[i, bootstrap_indices] = y_train_leaves[bootstrap_indices]
        return self

    def predict(self, X, quantile=None):
        """
        Predict regression value for X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.

        quantile : int, optional
            Value ranging from 0 to 100. By default, the mean is returned.

        check_input : boolean, (default=True)
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.

        Returns
        -------
        y : array of shape = [n_samples]
            If quantile is set to None, then return E(Y | X). Else return
            y such that F(Y=y | x) = quantile.
        """
        # apply method requires X to be of dtype np.float32
        X = check_array(X, dtype=np.float32, accept_sparse="csc")
        if quantile is None:
            return super(BaseForestQuantileRegressor, self).predict(X)

        sorter = np.argsort(self.y_train_)
        X_leaves = self.apply(X)
        weights = np.zeros((X.shape[0], len(self.y_train_)))
        quantiles = np.zeros((X.shape[0]))
        for i, x_leaf in enumerate(X_leaves):
            mask = self.y_train_leaves_ != np.expand_dims(x_leaf, 1)
            x_weights = ma.masked_array(self.y_weights_, mask)
            weights = x_weights.sum(axis=0)
            quantiles[i] = weighted_percentile(
                self.y_train_, quantile, weights, sorter)
        return quantiles


class RandomForestQuantileRegressor(BaseForestQuantileRegressor):
    """
    A random forest regressor that provides quantile estimates.

    A random forest is a meta estimator that fits a number of classifying
    decision trees on various sub-samples of the dataset and use averaging
    to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original
    input sample size but the samples are drawn with replacement if
    `bootstrap=True` (default).

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        whether to use out-of-bag samples to estimate
        the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    Attributes
    ----------
    estimators_ : list of DecisionTreeQuantileRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

    y_train_ : array-like, shape=(n_samples,)
        Cache the target values at fit time.

    y_weights_ : array-like, shape=(n_estimators, n_samples)
        y_weights_[i, j] is the weight given to sample ``j` while
        estimator ``i`` is fit. If bootstrap is set to True, this
        reduces to a 2-D array of ones.

    y_train_leaves_ : array-like, shape=(n_estimators, n_samples)
        y_train_leaves_[i, j] provides the leaf node that y_train_[i]
        ends up when estimator j is fit. If y_train_[i] is given
        a weight of zero when estimator j is fit, then the value is -1.

    References
    ----------
    .. [1] Nicolai Meinshausen, Quantile Regression Forests
        http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
    """
    def __init__(self,
                 n_estimators=10,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(RandomForestQuantileRegressor, self).__init__(
            base_estimator=DecisionTreeQuantileRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes


class ExtraTreesQuantileRegressor(BaseForestQuantileRegressor):
    """
    An extra-trees regressor that provides quantile estimates.

    This class implements a meta estimator that fits a number of
    randomized decision trees (a.k.a. extra-trees) on various sub-samples
    of the dataset and use averaging to improve the predictive accuracy
    and control over-fitting.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the forest.

    criterion : string, optional (default="mse")
        The function to measure the quality of a split. Supported criteria
        are "mse" for the mean squared error, which is equal to variance
        reduction as feature selection criterion, and "mae" for the mean
        absolute error.
        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered at each
          split.
        - If "auto", then `max_features=n_features`.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a percentage and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node:
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a percentage and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
        .. versionchanged:: 0.18
           Added float values for percentages.

    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    bootstrap : boolean, optional (default=False)
        Whether bootstrap samples are used when building trees.

    oob_score : bool, optional (default=False)
        Whether to use out-of-bag samples to estimate the R^2 on unseen data.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.

    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.

    Attributes
    ----------
    estimators_ : list of ExtraTreeQuantileRegressor
        The collection of fitted sub-estimators.

    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).

    n_features_ : int
        The number of features when ``fit`` is performed.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_prediction_ : array of shape = [n_samples]
        Prediction computed with out-of-bag estimate on the training set.

    y_train_ : array-like, shape=(n_samples,)
        Cache the target values at fit time.

    y_weights_ : array-like, shape=(n_estimators, n_samples)
        y_weights_[i, j] is the weight given to sample ``j` while
        estimator ``i`` is fit. If bootstrap is set to True, this
        reduces to a 2-D array of ones.

    y_train_leaves_ : array-like, shape=(n_estimators, n_samples)
        y_train_leaves_[i, j] provides the leaf node that y_train_[i]
        ends up when estimator j is fit. If y_train_[i] is given
        a weight of zero when estimator j is fit, then the value is -1.

    References
    ----------
    .. [1] Nicolai Meinshausen, Quantile Regression Forests
        http://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
    """
    def __init__(self,
                 n_estimators=10,
                 criterion='mse',
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0,
                 max_features='auto',
                 max_leaf_nodes=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False):
        super(ExtraTreesQuantileRegressor, self).__init__(
            base_estimator=ExtraTreeQuantileRegressor(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
       import os

from src.d00_conf.conf import conf, conf_loader
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

conf_loader("OEG")


def creating_string_bands_available(x):
    output = '_'.join([ele for ele in  x.available_temp if type(ele)==str])
    return output


def adding_busy_hour_data(df):
    df.loc[df['cell_tech'] == '2G', 'cell_occupation_dl_percentage_bh'] = conf["UPGRADE_SELECTION"][
                                                                              "BUSY_HOUR_RATE_2G"] * df.loc[df[
                                                                                                                'cell_tech'] == '2G', 'cell_occupation_dl_percentage']
    df.loc[df['cell_tech'] == '3G', 'cell_occupation_dl_percentage_bh'] = conf["UPGRADE_SELECTION"][
                                                                              "BUSY_HOUR_RATE_3G"] * df.loc[df[
                                                                                                                'cell_tech'] == '3G', 'cell_occupation_dl_percentage']
    df.loc[df['cell_tech'] == '4G', 'cell_occupation_dl_percentage_bh'] = conf["UPGRADE_SELECTION"][
                                                                              "BUSY_HOUR_RATE_4G"] * df.loc[df[
                                                                                                                'cell_tech'] == '4G', 'cell_occupation_dl_percentage']

    return df


def processing_codification_code_site(df_code_site, col_bands):
    for col in col_bands:
        df_bands = df_code_site.groupby('site_id')[col].unique().reset_index()
        df_bands = df_bands.rename(columns={col: 'available_temp'})
        df_bands['available_temp'] = df_bands['available_temp'].apply(lambda x: list(x))
        df_bands[col + '_available'] = df_bands.apply(creating_string_bands_available, axis=1)
        df_bands = df_bands.drop(columns='available_temp')
        df_code_site = df_code_site.merge(df_bands, on='site_id', how='left')
        # df_code_site['cell_name'] = df_code_site.cell_name.apply(lambda x: str(x).upper())
        df_code_site[col + '_available'] = df_code_site[col + "_available"].apply(lambda x: str(x).upper())
    df_code_site = df_code_site.drop_duplicates(subset=["site_id"])
    return df_code_site[["site_id"] + [col + "_available" for col in col_bands]]


def create_site_features(df):
    df = df.groupby(['date', 'year', 'week', 'week_period', 'site_id', 'cell_tech']).agg(
        {'cell_occupation_dl_percentage_bh': np.max,
         'code_utilization': np.max,
         'power_congestion': np.max  # 'BW_required':np.sum
         }).reset_index()
    df = df.rename(columns={'cell_tech': 'site_tech'})

    return df


def detect_site_congestion(x):
    if (x.site_tech == "3G") and ((x.code_utilization > 85) or (x.power_congestion > 85)):
        return 'CONGESTION_3G'
    elif (x.site_tech == "4G") and (x.cell_occupation_dl_percentage_bh > 85):
        return 'CONGESTION_4G'
    else:
        return 'NO_CONGESTION'


def getBand_3G(df):
    """
    - Lambda function to get all band 3G only of a site .
    """
    list_band = df.band_BW_available.split("_")
    band_value = []
    for band in list_band:
        for value in ["F1", "F2", "F3", "U1", "U2"]:
            if (value in band):
                band_value.append(value)

    return "_".join(sorted(band_value))


def getBand_4G(df):
    list_band = df.band_BW_available.split("_")

    value_str = []
    for band in list_band:
        for value in ["LTE1800", "LTE2100", "LTE900"]:

            if (value in band):
                value_str.append(value + band[-3:-1])

    return "_".join(sorted(value_str))


def getBandWithTech(df):
    if (df.site_tech == "3G"):
        return getBand_3G(df)
    if (df.site_tech == "4G"):
        return getBand_4G(df)
    return None


def select_band(x):
    if x.congestion == 'CONGESTION_3G':
        # F1 +F2
        if ('F1' in x.BW_available and 'F2' in x.BW_available and
                "F3" not in x.BW_available and "U1" not in x.BW_available and "U2" not in x.BW_available):
            # Todo
            # LTE = 5MHZ in 2100
            if ('LTE21005M' in x.band_BW_available):
                if (x.refarming_2G == 1):
                    x.bands_upgraded = "U900"
                else:
                    x.bands_upgraded = "new_site"
            else:
                x.bands_upgraded = "F3"

        # F1+F2+F3
        if (
                'F1' in x.BW_available and 'F2' in x.BW_available and "F3" in x.BW_available and "U1" not in x.BW_available and "U2" not in x.BW_available):
            if (x.refarming_2G == 1):
                x.bands_upgraded = "U900"
            else:
                x.bands_upgraded = "new_site"

        # F1+F2+F3+U1
        if ('F1' in x.BW_available and 'F2' in x.BW_available and
                "F3" in x.BW_available and "U1" in x.BW_available and "U2" not in x.BW_available):

            if ('LTE9005M' in x.band_BW_available):
                x.bands_upgraded = "new_site"
            elif (x.refarming_2G == 1):
                x.bands_upgraded = "U900"
            else:
                x.bands_upgraded = "new_site"

        # F1+F2 +U1
        if ('F1' in x.BW_available and 'F2' in x.BW_available and
                "F3" not in x.BW_available and "U1" in x.BW_available and "U2" not in x.BW_available):

            if ('LTE21005M' in x.band_BW_available):
                if ('LTE9005M' in x.band_BW_available):
                    x.bands_upgraded = "new_site"
                elif (x.refarming_2G == 1):
                    x.bands_upgraded = "U900"
                else:
                    x.bands_upgraded = "new_site"
            else:
                x.bands_upgraded = "F3"

        # F1+F2+U1+U2
        if ('F1' in x.BW_available and 'F2' in x.BW_available and
                "F3" not in x.BW_available and "U1" in x.BW_available and "U2" in x.BW_available):

            if ('LTE21005M' in x.band_BW_available):
                x.bands_upgraded = "new_site"
            else:
                x.bands_upgraded = "F3"

        # F1+U1
        if ('F1' in x.BW_available and 'F2' not in x.BW_available and
                "F3" not in x.BW_available and "U1" in x.BW_available and "U2" not in x.BW_available):
            if ('LTE210010M' in x.band_BW_available):
                if ('LTE9005M' in x.band_BW_available):
                    x.bands_upgraded = "new_site"
                elif (x.refarming_2G == 1):
                    x.bands_upgraded = "U900"
                else:
                    x.bands_upgraded = "new_site"
            else:
                x.bands_upgraded = "F2"

        # F1 + F2 + F3 + U1 + U2
        if ('F1' in x.BW_available and 'F2' in x.BW_available and
                "F3" in x.BW_available and "U1" in x.BW_available and "U2" in x.BW_available):
            x.bands_upgraded = "new_site"
        # # U1
        if ('F1' not in x.BW_available and 'F2' not in x.BW_available and
                "F3" not in x.BW_available and "U1" in x.BW_available and "U2" not in x.BW_available):
            if ("LTE2100" in x.band_available):

                if ("LTE900" in x.band_available):
                    x.bands_upgraded = "new_site"
                elif (x.refarming_2G == 1):

                    x.bands_upgraded = "U900"
                else:
                    x.bands_upgraded = "new_site"
            else:
                x.bands_upgraded = "F1_F2"

            x.bands_upgraded = "U1"
        # # F1
        if ('F1' in x.BW_available and 'F2' not in x.BW_available and
                "F3" not in x.BW_available and "U1" not in x.BW_available and "U2" not in x.BW_available):
            if ("LTE210010M" in x.band_BW_available):
                if (x.refarming_2G == 1):

                    x.bands_upgraded = "U900"
                else:
                    x.bands_upgraded = "new_site"
            else:
                x.bands_upgraded = "F2"

        # # F1+U1+U2
        if ('F1' in x.BW_available and 'F2' not in x.BW_available and
                "F3" not in x.BW_available and "U1" in x.BW_available and "U2" in x.BW_available):

            if ("LTE210010M" in x.band_BW_available):
                x.bands_upgraded = "new_site"
            else:
                x.bands_upgraded = "F2"

        # # U1+F1+F3 # mistake and will be cleaned
        if ('F1' in x.BW_available and 'F2' not in x.BW_available and
                "F3" in x.BW_available and "U1" in x.BW_available and "U2" not in x.BW_available):
            x.bands_upgraded = None

    if x.congestion == 'CONGESTION_4G':
        # 15M L1800
        if ('LTE1800' in x.band_available and 'LTE900' not in x.band_available and
                "LTE2100" not in x.band_available):
            # Todo
            # vérifier le refarming
            if (x.refarming_3G == 1):
                x.bands_upgraded = '5M_LTE2100'
            elif (x.refarming_2G == 1):
                x.bands_upgraded = '5M_LTE2100'
            else:
                x.bands_upgraded = '2600TDD/newsite'
        # 15 + 5(L1800 + L2100)
        if ('LTE1800' in x.band_available and 'LTE2100' in x.band_available and
                "LTE900" not in x.band_available and 'LTE21005M' in x.band_BW_available):
            if (x.refarming_3G == 1):
                x.bands_upgraded = '5M_LTE2100'
            elif (x.refarming_2G == 1):
                x.bands_upgraded = '5M_LTE2100'
            else:
                x.bands_upgraded = '2600TDD/newsite'

        # 15+5 (L1800+L900)
        if ('LTE1800' in x.band_available and 'LTE900' in x.band_available and
                "LTE2100" not in x.band_available):
            if (x.refarming_3G == 1):
                x.bands_upgraded = '5M_LTE2100'
            else:
                x.bands_upgraded = '2600TDD/newsite'

        # 15+10 (L1800+L2100)
        if ('LTE1800' in x.band_available and 'LTE2100' in x.band_available and
                "LTE900" not in x.band_available and "210010M" in x.band_BW_available):
            if (x.refarming_2G == 1):
                x.bands_upgraded = x.bands_upgraded + '5M_LTE900'
            else:
                x.bands_upgraded = x.bands_upgraded + '2600TDD/newsite'

        # 15+5+5 (L1800+L2100+L900)
        if ('LTE1800' in x.band_available and 'LTE2100' in x.band_available and
                "LTE900" in x.band_available and "LTE21005M" in x.band_BW_available):
            if (x.refarming_3G == 1):
                x.bands_upgraded = '5M_LTE2100'
            else:
                x.bands_upgraded = '2600TDD/newsite'
        # 15+10+5 (L1800+L2100+L900)

        if ('LTE1800' in x.band_available and 'LTE2100' in x.band_available and
                "LTE900" in x.band_available and "LTE210010M" in x.band_BW_available):
            x.bands_upgraded = '2600TDD/newsite'

        # #  15+15 (L1800 +L2100)
        # if ('LTE1800' in x.band_available and 'LTE2100' in x.band_available and
        #         "LTE900" not in  x.band_available and "LTE210015M" in x.band_BW_available):
        #     x.bands_upgraded="LTE210015M"
        #
        # if ('LTE1800' not in x.band_available and 'LTE2100' in x.band_available and
        #         "LTE900" not in  x.band_available and "LTE21005M" in x.band_BW_available):
        #     x.bands_upgraded="LTE21005M"

    return x


def select_band_with_densification(x):
    if x.congestion == 'CONGESTION_3G':
        # F1 +F2
        if ('F1' in x.BW_available and 'F2' in x.BW_available and
                "F3" not in x.BW_available and "U1" not in x.BW_available and "U2" not in x.BW_available):
            # Todo
            # LTE = 5MHZ in 2100
            if ('LTE21005M' in x.band_BW_available):
                if (x.refarming_2G == 1):
                    x.bands_upgraded = "U900"
                else:
                    x.bands_upgraded = "new_site"
            else:
                x.bands_upgraded = "F3"

        # F1+F2+F3
        if (
                'F1' in x.BW_available and 'F2' in x.BW_available and "F3" in x.BW_available and "U1" not in x.BW_available and "U2" not in x.BW_available):
            if (x.refarming_2G == 1):
                x.bands_upgraded = "U900"
            else:
                x.bands_upgraded = "new_site"

        # F1+F2+F3+U1
        if ('F1' in x.BW_available and 'F2' in x.BW_available and
                "F3" in x.BW_available and "U1" in x.BW_available and "U2" not in x.BW_available):

            if ('LTE9005M' in x.band_BW_available):
                x.bands_upgraded = "new_site"
            elif (x.refarming_2G == 1):
                x.bands_upgraded = "U900"
            else:
                x.bands_upgraded = "new_site"

        # F1+F2 +U1
        if ('F1' in x.BW_available and 'F2' in x.BW_available and
                "F3" not in x.BW_available and "U1" in x.BW_available and "U2" not in x.BW_available):

            if ('LTE21005M' in x.band_BW_available):
                if ('LTE9005M' in x.band_BW_available):
                    x.bands_upgraded = "new_site"
                elif (x.refarming_2G == 1):
                    x.bands_upgraded = "U900"
                else:
                    x.bands_upgraded = "new_site"
            else:
                x.bands_upgraded = "F3"

        # F1+F2+U1+U2
        if ('F1' in x.BW_available and 'F2' in x.BW_available and
                "F3" not in x.BW_available and "U1" in x.BW_available and "U2" in x.BW_available):

            if ('LTE21005M' in x.band_BW_available):
                x.bands_upgraded = "new_site"
            else:
                x.bands_upgraded = "F3"

        # F1+U1
        if ('F1' in x.BW_available and 'F2' not in x.BW_available and
                "F3" not in x.BW_available and "U1" in x.BW_available and "U2" not in x.BW_available):
            if ('LTE210010M' in x.band_BW_available):
                if ('LTE9005M' in x.band_BW_available):
                    x.bands_upgraded = "new_site"
                elif (x.refarming_2G == 1):
                    x.bands_upgraded = "U900"
                else:
                    x.bands_upgraded = "new_site"
            else:
                x.bands_upgraded = "F2"

        # F1 + F2 + F3 + U1 + U2
        if ('F1' in x.BW_available and 'F2' in x.BW_available and
                "F3" in x.BW_available and "U1" in x.BW_available and "U2" in x.BW_available):
            x.bands_upgraded = "new_site"

    if x.congestion == 'CONGESTION_4G':

        # 15M L1800
        bw_required = x.BW_required

        while (bw_required >= 5 & x.newsite == 0):

            # 15M L1800
            if ('LTE1800' in x.band_available and 'LTE900' not in x.band_available and
                    "LTE2100" in x.band_available):
                # Todo
                # vérifier le refarming
                if (x.refarming_3G == 1):

                    x.bands_upgraded = x.bands_upgraded + '5M_LTE2100'
                    x.band_available = x.band_available + '_LTE2100'
                    bw_required = bw_required - 5
                    print(x.bands_upgraded + '5M_LTE2100')

                elif (x.refarming_2G == 1):
                    x.bands_upgraded = x.bands_upgraded + '5M_LTE2100'
                    x.band_available = x.band_available + '_LTE2100'
                    bw_required = bw_required - 5
                    print(x.bands_upgraded + '5M_LTE2100')
                else:
                    x.bands_upgraded = x.bands_upgraded + '_2600TDD/newsite'
                    x.newsite = 1
                print(x.bands_upgraded, x.newsite)
            # 15 + 5(L1800 + L2100)
            elif ('LTE1800' in x.band_available and 'LTE2100' not in x.band_available and
                  "LTE900" not in x.band_available and 'LTE21005M' in x.band_BW_available):
                if (x.refarming_3G == 1):
                    x.bands_upgraded = x.bands_upgraded + '5M_LTE2100'
                    x.band_available = x.band_available + '_LTE2100'
                    bw_required = bw_required - 5
                    print(x.bands_upgraded + '5M_LTE2100')
                elif (x.refarming_2G == 1):
                    x.bands_upgraded = x.bands_upgraded + '5M_LTE2100'
                    x.band_available = x.band_available + '_LTE2100'
                    bw_required = bw_required - 5
                    print(x.bands_upgraded + '5M_LTE2100')
                else:
                    x.bands_upgraded = x.bands_upgraded + '_2600TDD/newsite'
                    x.newsite = 1

            # 15+5 (L1800+L900)
            elif ('LTE1800' in x.band_available and 'LTE900' in x.band_available and
                  "LTE2100" in x.band_available):
                if (x.refarming_3G == 1):
                    x.bands_upgraded = x.bands_upgraded + '5M_LTE2100'
                    x.band_available = x.band_available + '_LTE2100'
                    bw_required = bw_required - 5
                    print(x.bands_upgraded + '5M_LTE2100')
                else:
                    x.bands_upgraded = x.bands_upgraded + '_2600TDD/newsite'
                    x.newsite = 1

            # 15+10 (L1800+L2100)
            elif ('LTE1800' in x.band_available and 'LTE2100' in x.band_available and
                  "LTE900" not in x.band_available and "210010M" in x.band_BW_available):
                if (x.refarming_2G == 1):
                    x.bands_upgraded = x.bands_upgraded + '5M_LTE900'
                    x.band_available = x.band_available + '_LTE900'
                    bw_required = bw_required - 5
                    print(x.bands_upgraded + '5M_LTE900')

                else:
                    x.bands_upgraded = x.bands_upgraded + '_2600TDD/newsite'
                    x.newsite = 1

            # 15+5+5 (L1800+L2100+L900)
            elif ('LTE1800' in x.band_available and 'LTE2100' in x.band_available and
                  "LTE900" in x.band_available and "LTE21005M" in x.band_BW_available):
                if (x.refarming_3G == 1):
                    x.bands_upgraded = x.bands_upgraded + '5M_LTE2100'
                    x.band_available = x.band_available + '_LTE2100'
                    bw_required = bw_required - 5
                else:
                    x.bands_upgraded = x.bands_upgraded + '_2600TDD/newsite'
                    x.newsite = 1

            # 15+10+5 (L1800+L2100+L900)
            elif ('LTE1800' in x.band_available and 'LTE2100' in x.band_available and
                  "LTE900" in x.band_available and "LTE210010M" in x.band_BW_available):

                x.bands_upgraded = x.bands_upgraded + '_2600TDD/newsite'
                x.newsite = 1
            else:
                x.bands_upgraded = x.bands_upgraded + "NO_ACTION"
                break

    return x


def create_refarming_features(df):
    df_3G = df[df.site_tech == "3G"].groupby(['week_period', 'site_id'])[
        ["code_utilization", "power_congestion"]].first()
    df_3G.reset_index(inplace=True)
    df_3G.rename(columns={"code_utilization": "code_utilization_3G", \
                          "power_congestion": "power_congestion_3G"},
                 inplace=True)

    df_2G = df[df.site_tech == "2G"].groupby(['week_period',\
                        'site_id'])[["cell_occupation_dl_percentage_bh"]].first()
    df_2G.reset_index(inplace=True)
    df_2G.rename(columns={"cell_occupation_dl_percentage_bh":\
                              "cell_occupation_dl_percentage_bh_2G"}, inplace=True)

    df = df.merge(df_3G, how="left", on=['week_period', 'site_id'])
    df = df.merge(df_2G, how="left", on=['week_period', 'site_id'])
    ## creating 3G refarming code:
    df["refarming_3G"] = 0
    df.loc[(df['code_utilization_3G'] < 90) & (df['power_congestion_3G'] < 90), "refarming_3G"] = 1
    ## creating 2G refarming code:
    df["refarming_2G"] = 0
    df.loc[(df['cell_occupation_dl_percentage_bh_2G'] < 2), "refarming_2G"] = 1
    return df


def compute_the_band_selected_for_the_upgrade(df, week_of_the_upgrade):
    df['week_of_the_upgrade'] = week_of_the_upgrade
    df.week_period = df.week_period.apply(lambda x: str(x))
    df = df[~df.bands_upgraded.isnull()]
    output = df.loc[df['week_period'] == week_of_the_upgrade][
        ['site_id', 'week_of_the_upgrade', 'band_before', 'bands_upgraded', 'site_tech']]
    output.to_csv(os.path.join(conf["PATH"]["MODELS_OUTPUT"], 'upgrade_selection_output', 'df_for_week_upgrade.csv'),
                  sep="|", index=False)
    df = df.drop_duplicates(subset=['site_id', 'congestion', 'week_of_the_upgrade', 'bands_upgraded', 'site_tech'],
                            keep='first')
    df_upgraded_band = df[['site_id', 'site_tech',
                           'week_period', 'congestion',
                           'band_before', 'bands_upgraded']]
    df_upgraded_band.to_csv(
        os.path.join(conf["PATH"]["MODELS_OUTPUT"], 'upgrade_selection_output', 'df_all_upgrade.csv'), sep="|",
        index=False)

    return output



# ----------------------------
def get_cell_tech(x):
    if x.startswith("UMTS"):
        return "3G"
    if x.startswith("LTE"):
        return "4G"
    if x.startswith("GSM"):
        return "2G"
    return get_cell_tech

def fill_cell_tech(x):
    techs = []
    if x["cell_tech_available"] == "":
        if "LTE" in x["band_available"]:
            techs.append("4G")
        if "UMTS" in x["band_available"]:
            techs.append("3G")
        if "GSM" in x["band_available"]:
            techs.append("2G")
        x["cell_tech_available"] =  '_'.join(techs)
    return x

def upgrade_selection_pipeline(df, df_site, week_of_the_upgrade=conf['TRAFFIC_IMPROVEMENT']['WEEK_OF_THE_UPGRADE']):
    df_site = df_site[['site_id', 'site_band_width', 'cell_id', 'cell_band']]
    df_site.rename(columns={"site_band_width": "BW", "cell_band": "band"}, inplace=True)
    df_cell_tech = df.drop_duplicates(subset=["site_id", "cell_tech"]).reset_index(drop=True)
    df_site = df_site.merge(df_cell_tech[["site_id", "cell_tech"]], on="site_id", how="left").fillna('')
    df_site["band_BW"] = df_site["band"] + df_site["BW"]

    df_code_site = processing_codification_code_site(df_site, ["BW", "band", "band_BW", "cell_tech"])

    df_adding_bh = adding_busy_hour_data(df)

    ## add busy hour KPI
    df_congestion = create_site_features(df_adding_bh)
    ## detect congestion
    print(df_congestion.site_tech.unique())
    df_congestion['congestion'] = df_congestion.apply(detect_site_congestion, axis=1)

    ## creating refarming features
    df_congestion = create_refarming_features(df_congestion)
    df_congestion = df_congestion[(df_congestion.congestion != 'NO_CONGESTION') | (
            (df_congestion.congestion.isnull() == False) & (df_congestion.congestion != 'NO_CONGESTION'))]
    df_congestion = df_congestion.merge(df_code_site, how="left", on='site_id')
    df_congestion["band_before"] = df_congestion.apply(getBandWithTech, axis=1)
    df_congestion["bands_upgraded"] = ''
    ## Select Band
    df_congestion = df_congestion.apply(select_band, axis=1)
    df_congestion.to_csv(
        os.path.join(conf["PATH"]["MODELS_OUTPUT"], 'upgrade_selection_output', 'df_site_congestion.csv'), index=False,
        sep="|")
    selected_band = compute_the_band_selected_for_the_upgrade(df_congestion, week_of_the_upgrade=week_of_the_upgrade)
    return selected_band

# cells_id = ['0002EB7370AD1C384E4FDAACFF046D02',
#             '000314FBF56860DB479290AF6D1379D6',
#             '00032D7DD61A4CCA0523E95F24AA2F0D',
#             '00057581D335027A8C35C3BFDAB09EE0',
#             '0007D51D2DF5A7C3919223C4EE832EA2',
#
#             '00017AFE955DA76BFE7CCA3340CBABB0',
#             '0005B8BC18F216212A74AEC768DE210A',
#             '000638D83A19FE5EE1F54BF4B1C8DF31',
#             '0007B7389BC6FD8D19F1016851AE030D',
#             '00089EF52E71524D88F1508F1B3622BC']
#

def prepare_data_4g_bh_weekly(compute_data_4g_bh_weekly):
    if compute_data_4g_bh_weekly:
        df_4g_bh_daily = pd.read_csv( \
            "/home/sc_team/arwa.abdelhamid/data/01_raw/Requested_KPIs/Huawei Smart Capex/4G_BH_Daily.csv", \
            sep=",")
        df_4g_bh_selected_daily = df_4g_bh_daily[
            ["Global_Cell_Id", "Date", "DL PRB Usage Rate(%)", "DL PDCP SDU TRAFFIC VOLUME (GB)"]]
        df_4g_bh_selected_daily.columns = ["cell_id", "date", "prb_used_busy_hour", "traffic_data_busy_hour"]
        df_4g_bh_selected_daily["date"] = pd.to_datetime(df_4g_bh_selected_daily["date"])
        df_4g_bh_selected_daily["week_period"] = \
            df_4g_bh_selected_daily["date"].apply(lambda x: x.strftime("%Y%W"))

        df_4g_bh_selected_daily[["prb_used_busy_hour",
                                 "traffic_data_busy_hour"]] = \
            df_4g_bh_selected_daily[["prb_used_busy_hour",
                                     "traffic_data_busy_hour"]].astype(float)

        df_4g_bh_selected_weekly = \
            df_4g_bh_selected_daily.groupby(["cell_id", "week_period"]) \
                .agg({"prb_used_busy_hour": np.nanmean,
                      "traffic_data_busy_hour": np.nanmean})
        df_4g_bh_selected_weekly.reset_index(inplace=True)
        df_4g_bh_selected_weekly["cell_tech"] = "4G"
        df_4g_bh_selected_weekly.to_csv( \
            "/data/OEG/02_intermediate/work_mahmoud_test_v0/df_4g_bh_weekly.csv", sep="|", index=False)
    else:
        df_4g_bh_selected_weekly = pd.read_csv( \
            "/data/OEG/02_intermediate/work_mahmoud_test_v0/df_4g_bh_weekly.csv", sep="|")
    return df_4g_bh_selected_weekly


def prepare_data_3g_bh_weekly(compute_data_3g_bh_weekly):
    if compute_data_3g_bh_weekly:
        df_3g_bh_daily = pd.read_csv( \
            "/home/sc_team/arwa.abdelhamid/data/01_raw/Requested_KPIs/Huawei Smart Capex/3G_BH_Daily.csv", \
            sep=",")
        df_3g_bh_daily = df_3g_bh_daily[df_3g_bh_daily["Code Utilization"] != "NIL"]
        df_3g_bh_selected_daily = df_3g_bh_daily[
            ["Global_Cell_Id", "Date", "Code Utilization", \
             "power congestion RATE 2", "TOTAL DATA TRAFFIC - DL (GB)"]]
        df_3g_bh_selected_daily.columns = ["cell_id", "date", "code_utilisation", \
                                           "power_congestion", "traffic_data_busy_hour"]
        df_3g_bh_selected_daily["date"] = pd.to_datetime(df_3g_bh_selected_daily["date"])
        df_3g_bh_selected_daily["week_period"] = \
            df_3g_bh_selected_daily["date"].apply(lambda x: x.strftime("%Y%W"))
        df_3g_bh_selected_daily[["code_utilisation",
                                 "power_congestion",
                                 "traffic_data_busy_hour"]] = \
            df_3g_bh_selected_daily[["code_utilisation",
                                     "power_congestion",
                                     "traffic_data_busy_hour"]].astype(float)
        df_3g_bh_selected_daily = \
            df_3g_bh_selected_daily.groupby(["cell_id", "week_period"]) \
                .agg({"code_utilisation": np.nanmean,
                      "power_congestion": np.nanmean,
                      "traffic_data_busy_hour": np.nanmean})
        df_3g_bh_selected_daily.reset_index(inplace=True)
        df_3g_bh_selected_daily["cell_tech"] = "3G"
        df_3g_bh_selected_daily.to_csv( \
            "/data/OEG/02_intermediate/work_mahmoud_test_v0/df_3g_bh_weekly.csv",
            sep="|", index=False)
    else:
        df_3g_bh_selected_weekly = pd.read_csv( \
            "/data/OEG/02_intermediate/work_mahmoud_test_v0/df_3g_bh_weekly.csv", sep="|")
    return df_3g_bh_selected_weekly


def get_linear_regression_coefficient(x, y):
    model = LinearRegression(fit_intercept=False)
    model.fit(x, y)
    return model.coef_[0]


def select_data_for_one_cell(cell_id, df_historical, df_predicted, df_3g_bh, df_4g_bh):
    df_historical_one_cell = df_historical[df_historical["cell_id"] == cell_id]
    df_predicted_one_cell = df_predicted[df_predicted["cell_id"] == cell_id]
    cell_tech = df_historical_one_cell["cell_tech"].unique()[0]
    if cell_tech == "3G":
        df_bh_one_cell = df_3g_bh[df_3g_bh["cell_id"] == cell_id]
    if cell_tech == "4G":
        df_bh_one_cell = df_4g_bh[df_4g_bh["cell_id"] == cell_id]

    df_bh_one_cell["week_period"] = df_bh_one_cell["week_period"].astype(int)
    df_historical_one_cell["week_period"] = df_historical_one_cell["week_period"].astype(int)
    df_predicted_one_cell["week_period"] = df_predicted_one_cell["week_period"].astype(int)

    df_bh_one_cell.sort_values(by="week_period", ascending=True, inplace=True)
    df_historical_one_cell.sort_values(by="week_period", ascending=True, inplace=True)
    df_predicted_one_cell.sort_values(by="week_period", ascending=True, inplace=True)
    return df_historical_one_cell, df_predicted_one_cell, df_bh_one_cell, cell_tech


def compute_predicted_traffic_busy_hour(df_historical_one_cell,
                                        df_predicted_one_cell,
                                        df_bh_one_cell,
                                        ):
    df_historical_one_cell['increase_factor_historical'] = \
        df_historical_one_cell['total_data_traffic_dl_gb'] / \
        df_historical_one_cell['total_data_traffic_dl_gb'].shift(1)
    traffic_data_last_week = df_historical_one_cell['total_data_traffic_dl_gb'].values[-1]
    df_predicted_one_cell["increase_factor_predicted"] = \
        df_predicted_one_cell["total_data_traffic_dl_gb"] / traffic_data_last_week
    df_predicted_one_cell['traffic_busy_hour_predicted'] = \
        df_bh_one_cell['traffic_data_busy_hour'].values[-1] * df_predicted_one_cell['increase_factor_predicted']
    return df_bh_one_cell, df_predicted_one_cell


def add_new_busy_hour_data(cell_tech, df_predicted_one_cell, df_bh_one_cell):
    if cell_tech == "3G":
        coef_code_utilisation = \
            get_linear_regression_coefficient(df_bh_one_cell["traffic_data_busy_hour"].values.reshape(-1, 1), \
                                              df_bh_one_cell["code_utilisation"].values.reshape(-1, 1))
        coef_power_congestion = \
            get_linear_regression_coefficient(df_bh_one_cell["traffic_data_busy_hour"].values.reshape(-1, 1), \
                                              df_bh_one_cell["power_congestion"].values.reshape(-1, 1))

        df_predicted_one_cell['power_congestion_busy_hour_predicted'] = \
            df_predicted_one_cell['traffic_busy_hour_predicted'] * coef_power_congestion
        df_predicted_one_cell['code_utilisation_busy_hour_predicted'] = \
            df_predicted_one_cell['traffic_busy_hour_predicted'] * coef_code_utilisation

    if cell_tech == "4G":
        coef_prb_used = \
            get_linear_regression_coefficient(df_bh_one_cell["traffic_data_busy_hour"].values.reshape(-1, 1), \
                                              df_bh_one_cell["prb_used_busy_hour"].values.reshape(-1, 1))
        df_predicted_one_cell['prb_used_busy_hour_predicted'] = \
            df_predicted_one_cell['traffic_busy_hour_predicted'] * coef_prb_used

    return df_predicted_one_cell


def add_congestion_variable(x, df_site):
    if x["cell_tech"] == "3G":
        if x["power_congestion_busy_hour_predicted"] > 85 \
                or x["code_utilisation_busy_hour_predicted"] > 85:
            x["congestion"] = "CONGESTION_3G"
        else:
            x["congestion"] = "NO_CONGESTION"
    if x["cell_tech"] == "4G":
        bw = df_site[df_site["cell_id"] == cells_id[0]]["BW"].mode()[0][:-1]
        x["prb_available"] = float(bw) * 5
        if x["prb_used_busy_hour_predicted"] / x["prb_available"] > 0.85:
            x["congestion"] = "CONGESTION_4G"
        else:
            x["congestion"] = "NO_CONGESTION"
    return x


def get_nb_carriers(x):
    nb_carriers = max(round((x["power_congestion_busy_hour_predicted"] - 85) / 85, 0),
                      round((x["code_utilisation_busy_hour_predicted"] - 85) / 85, 0))
    return nb_carriers

def prepare_data_for_congestion_detection(df_historical, df_predicted,
                                          df_3g_bh, df_4g_bh, df_site):
    cells_id = df_predicted["cell_id"].unique()
    list_of_df_predicted = []
    for j, cell_id in enumerate(cells_id):
        print("run congestion detection")
        try:
            print(str(j), " out of ", len(cells_id))
            df_historical_one_cell, df_predicted_one_cell, \
            df_bh_one_cell, cell_tech = select_data_for_one_cell(cell_id, df_historical,
                                                                 df_predicted,
                                                                 df_3g_bh, df_4g_bh)
            df_bh_one_cell, df_predicted_one_cell = \
                compute_predicted_traffic_busy_hour(df_historical_one_cell,
                                                    df_predicted_one_cell,
                                                    df_bh_one_cell)
            df_predicted_one_cell = add_new_busy_hour_data(cell_tech,
                                                           df_predicted_one_cell,
                                                           df_bh_one_cell)
            df_predicted_one_cell = df_predicted_one_cell.apply(add_congestion_variable,
                                                                df_site=df_site,
                                                                axis=1)

            list_of_df_predicted.append(df_predicted_one_cell)
        except:
            continue
        if j%5000 == 0 and j!=0:
            df_predicted_with_congestion_sample = pd.concat(list_of_df_predicted)
            dir_ = "/data/OEG/02_intermediate/work_mahmoud_test_v0/intermdiate_congestion"
            name = "df_prediction_with_congestion"+str(j) + ".csv"
            df_predicted_with_congestion_sample.to_csv(os.path.join(dir_,name),sep="|",index=False)
            df_predicted_with_congestion_sample.reset_index(inplace=True, drop=True)

    df_predicted_with_congestion = pd.concat(list_of_df_predicted)
    df_predicted_with_congestion.reset_index(inplace=True, drop=True)
    return df_predicted_with_congestion

def get_congestion_by_sectors(df_predicted_with_congestion):
    df_predicted_with_congestion = \
        df_predicted_with_congestion[df_predicted_with_congestion["congestion"] \
                                     != "NO_CONGESTION"]
    df_predicted_with_congestion_4g = \
        df_predicted_with_congestion[df_predicted_with_congestion["congestion"] \
                                     == "CONGESTION_4G"]
    df_predicted_with_congestion_4g["BW_required"] = \
        df_predicted_with_congestion_4g["prb_used_busy_hour_predicted"] / 5

    df_congestion_sectors_4g = df_predicted_with_congestion_4g.groupby(["site_id", "cell_sector",
                                                                        "week_period",
                                                                        ])["BW_required"].sum()
    df_congestion_sectors_4g = df_congestion_sectors_4g.reset_index()
    df_congestion_sectors_4g = df_congestion_sectors_4g.groupby(["week_period", "site_id"]).apply( \
        lambda x: x.sort_values(by="BW_required", ascending=False).iloc[0, :])
    df_congestion_sectors_4g.reset_index(inplace=True, drop=True)

    df_predicted_with_congestion_3g = \
        df_predicted_with_congestion[df_predicted_with_congestion["congestion"] \
                                     == "CONGESTION_3G"]
    df_predicted_with_congestion_3g["additional_carriers_number"] = \
        df_predicted_with_congestion_3g.apply(get_nb_carriers, axis=1)
    df_congestion_sectors_3g = df_predicted_with_congestion_3g.groupby(["site_id", "cell_sector",
                                                                        "week_period",
                                                                        ])["additional_carriers_number"].sum()
    df_congestion_sectors_3g = df_congestion_sectors_3g.reset_index()
    df_congestion_sectors_3g = df_congestion_sectors_3g.groupby(["week_period", "site_id"]).apply( \
        lambda x: x.sort_values(by="additional_carriers_number", ascending=False).iloc[0, :])
    df_congestion_sectors_3g.reset_index(inplace=True, drop=True)

    df_congestion_sectors_3g["week_period"] = \
        df_congestion_sectors_3g["week_period"].astype(int)
    df_congestion_sectors_4g["week_period"] = \
        df_congestion_sectors_4g["week_period"].astype(int)
    #df_congestion_sectors_4g.sort_values(by="week_period",inplace=True,ascending=True)
    #df_congestion_sectors_3g.sort_values(by="week_period", inplace=True, ascending=True)
    df_congestion_sectors_4g = \
        df_congestion_sectors_4g[int(conf["TRAFFIC_IMPROVEMENT"]["WEEK_OF_THE_UPGRADE"]) \
    > df_congestion_sectors_4g["week_period"]]
    df_congestion_sectors_3g = \
        df_congestion_sectors_3g[int(conf["TRAFFIC_IMPROVEMENT"]["WEEK_OF_THE_UPGRADE"]) \
    > df_congestion_sectors_3g["week_period"]]
    gps_4g = df_congestion_sectors_4g.groupby(["site_id"])
    gps_4g = [gp[1] for gp in gps_4g]
    df_congestion_site_4g = \
        pd.concat( [get_first_congestion(gp,tech="4G") for gp in gps_4g])
    gps_3g = df_congestion_sectors_3g.groupby(["site_id"])
    gps_3g = [gp[1] for gp in gps_3g]
    df_congestion_site_3g = \
        pd.concat([get_first_congestion(gp, tech="3G") for gp in gps_3g])

    return df_congestion_site_3g, df_congestion_site_4g

def get_first_congestion(df,tech):
    site_id = df["site_id"].unique()[0]
    min_week_period = df["week_period"].min()
    sector = df["cell_sector"][df["week_period"] == min_week_period].values[0]
    if tech == "4G":
        kpi_value =  df["BW_required"][df["week_period"] == min_week_period].values[0]
        kpi_name = "BW_required"
    if tech == "3G":
        kpi_value = df["additional_carriers_number"][\
            df["week_period"] == min_week_period].values[0]
        kpi_name = "additional_carriers_number"
    columns = ["site_id","week_period","site_sector",kpi_name]
    df_congestion_site = pd.DataFrame(
        np.array([site_id,min_week_period,sector,kpi_value]).reshape(1,len(columns)),
                                      columns = columns)
    return df_congestion_site


def upgrade_selection_pipeline(df_predicted_traffic_kpis, df_sites):
    df_sites = pd.read_csv( \
        "/data/OEG/02_intermediate/work_mahmoud_test_v0/df_sites_with_bw_v0.csv", sep="|")
    df_cell_sector_mapping = pd.read_csv("/data/OEG/01_raw/deployment_history/cell_sector_mapping.csv", sep=",")
    df_cell_sector_mapping.columns = ["site_id", "cell_sector", "cell_id"]
    df_sites = pd.merge(left=df_sites,
                        right=df_cell_sector_mapping,
                        left_on=["site_id", "cell_id"],
                        right_on=["site_id", "cell_id"],
                        how="left")
    df_traffic_weekly_kpis = pd.read_csv("/data/OEG/02_intermediate/processed_oss_all.csv",
                                         sep="|")
    df_predicted_traffic_kpis = pd.read_csv(
        "/data/OEG/02_intermediate/new_traffic_forecasting_boxcox/df_predicted_traffic_kpis.csv",
        sep="|")

    df_traffic_weekly_kpis = pd.merge(left=df_traffic_weekly_kpis,
                                      right=df_sites[["cell_id", "site_id",
                                                      "cell_band", "site_region"]],
                                      on="cell_id",
                                      how="left")
    df_predicted_traffic_kpis = pd.merge(left=df_predicted_traffic_kpis,
                                         right=df_sites[["cell_id", "site_region",
                                                         "cell_sector"]],
                                         on="cell_id",
                                         how="left")

    df_3g_bh = prepare_data_3g_bh_weekly(compute_data_3g_bh_weekly=False)
    df_4g_bh = prepare_data_4g_bh_weekly(compute_data_4g_bh_weekly=False)
    df_predicted_with_congestion = prepare_data_for_congestion_detection(
        df_traffic_weekly_kpis,
        df_predicted_traffic_kpis,
        df_3g_bh,
        df_4g_bh,
        df_sites)

    df_congestion_site_3g, df_congestion_site_4g = \
        get_congestion_by_sectors(df_predicted_with_congestion)
    df_congestion_site_3g["site_tech"] = "3G"
    df_congestion_site_3g["congestion"] = 'CONGESTION_3G'
    df_congestion_site_4g["site_tech"] = "4G"
    df_congestion_site_4g["congestion"] = 'CONGESTION_4G'

    df_sites = pd.read_csv( \
        "/data/OEG/02_intermediate/work_mahmoud_test_v0/df_sites_with_bw_v0.csv",
        sep="|")
    df_predicted_traffic_kpis = pd.read_csv(
        "/data/OEG/02_intermediate/new_traffic_forecasting_boxcox/df_predicted_traffic_kpis.csv",
        sep="|")
    df_congestion_site_3g = pd.read_csv( \
        "/data/OEG/02_intermediate/work_mahmoud_test_v0/df_congestion_site_3g.csv",
        sep="|")
    df_congestion_site_4g = pd.read_csv( \
        "/data/OEG/02_intermediate/work_mahmoud_test_v0/df_congestion_site_4g.csv",
        sep="|")

    df_cell_tech_bands = df_predicted_traffic_kpis[["cell_id", "cell_band"]].drop_duplicates()
    cells_to_handle = df_cell_tech_bands["cell_id"].unique().tolist()
    df_sites = df_sites[df_sites["cell_id"].isin(cells_to_handle)]
    df_sites.dropna(inplace=True, subset=["cell_band"])

    df_sites["cell_tech"] = ""
    df_sites["cell_tech"] = df_sites["cell_band"].apply(get_cell_tech)
    df_sites = df_sites[['site_id', 'site_band_width', 'cell_id', 'cell_band']]
    df_sites.rename(columns={"site_band_width": "BW", "cell_band": "band"}, inplace=True)
    df_cell_tech = df_predicted_traffic_kpis.drop_duplicates( \
        subset=["site_id", "cell_tech"]).reset_index(drop=True)[["site_id", "cell_tech"]]
    df_sites = df_sites.merge(df_cell_tech,
                              on="site_id", how="left")
    df_sites["band_BW"] = df_sites["band"] + df_sites["BW"]
    df_code_site = processing_codification_code_site(df_sites, ["BW", "band",
                                                                "band_BW", "cell_tech"])
    df_code_site = df_code_site.apply(fill_cell_tech, axis=1)
    df_congestion_site_3g["site_tech"] = "3G"
    df_congestion_site_3g["congestion"] = 'CONGESTION_3G'
    df_congestion_site_4g["site_tech"] = "4G"
    df_congestion_site_4g["congestion"] = 'CONGESTION_4G'
    df_congestion = pd.concat([df_congestion_site_3g, df_congestion_site_4g])
    df_congestion = df_congestion.merge(df_code_site, how="left", on='site_id')
    df_congestion["band_before"] = df_congestion.apply(getBandWithTech, axis=1)
    df_congestion["bands_upgraded"] = ''

    df_predicted_traffic_kpis_with_bh = adding_busy_hour_data(df_predicted_traffic_kpis)
    df_sites_features = create_site_features(df_predicted_traffic_kpis_with_bh)
    df_sites_with_refarming_features = create_refarming_features(df_sites_features)
    df_sites_with_refarming_features = \
        df_sites_with_refarming_features[["site_id", "week_period",
                                          "site_tech", "refarming_2G",
                                          "refarming_3G"]]

    df_congestion_with_refarming_features = pd.merge(left=df_congestion,
                                                     right=df_sites_with_refarming_features,
                                                     on=["site_id", "week_period", "site_tech"])

    ## Select Band
    max_week_period = df_traffic_weekly_kpis["week_period"].astype(int).max()
    df_congestion_with_added_bands = \
        df_congestion_with_refarming_features.apply(select_band, axis=1)
    df_congestion_with_added_bands = df_congestion_with_added_bands[ \
        df_congestion_with_added_bands["week_period"].astype(int) > max_week_period]

    return df_congestion_with_added_bands

