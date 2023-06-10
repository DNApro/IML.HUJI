from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        error = np.inf
        for j, sign in product(range(X.shape[1]), [-1, 1]):
            thr, thr_err = self._find_threshold(X[:, j], y, sign)
            if thr_err < error:
                self.threshold_, self.j_, self.sign_, error = thr, j, sign, thr_err

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] < self.threshold_, -self.sign_, self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # In class, we learned that a threshold classifier aims to find the maximum value that has a label of -1.
        # Sorting the input reduces the running time of finding that value from O(n^2) to O(n) instead.
        # After searching on Google, I found that numpy sorts an array inO(nlog(n)),
        # so the total running time is O(nlog(n)).
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        labels = labels[sorted_indices]
        abs_labels = np.abs(labels)

        # Calculating the Threshold-Based loss
        tb_loss = np.sum(abs_labels[np.sign(labels) != sign])
        # Calculating the Cumulative Threshold loss
        cumulative_loss = tb_loss + np.cumsum(sign * labels)

        tb_loss = np.append(tb_loss, cumulative_loss)
        th_ind = np.argmin(tb_loss)
        threshold_values = np.concatenate([[-np.inf], sorted_values[1:], [np.inf]])
        return threshold_values[th_ind], tb_loss[th_ind]
        # return np.concatenate([-np.inf], values[1:], [np.inf])[th_ind], tb_loss[th_ind]



    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from IMLearn.metrics.loss_functions import misclassification_error
        return misclassification_error(y, self._predict(X))




