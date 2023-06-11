from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    # Generate indices and split into folds
    divs = np.arange(X.shape[0])
    folds = np.array_split(divs, cv)
    train_err = 0.0
    test_err = 0.0

    for i, fold in enumerate(folds):
        # Exclude part from training
        train_X, test_X = np.delete(X, fold, axis=0), X[fold]
        train_y, test_y = np.delete(y, fold), y[fold]

        # Fit estimator to training data
        curr_estimator = deepcopy(estimator)
        curr_estimator.fit(train_X, train_y)


        # Calculate scores
        train_err += scoring(train_y, curr_estimator.predict(train_X))
        test_err += scoring(test_y, curr_estimator.predict(test_X))

    # averaging both errors
    return train_err / cv, test_err / cv
