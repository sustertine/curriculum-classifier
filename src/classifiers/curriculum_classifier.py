from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from typing import Callable, List, Tuple
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

from src.utils.utils import group_by_difficulty


class CurriculumClassifier(BaseEstimator, ClassifierMixin):
    """
    A classifier that trains using curriculum learning by splitting the dataset into groups based on learning difficulty.

    Parameters:
    -----------
    base_estimator : BaseEstimator
        The base estimator to be used for training.
    method : str, default='entropy'
        The method to calculate difficulty ('entropy' or 'avg_confidence').
    custom_method : Callable[[np.ndarray], np.ndarray], optional
        A custom method to calculate difficulty. If provided, it overrides the native methods.
    n_groups : int, default=3
        The number of groups to split the data into.
    """

    def __init__(self, base_estimator: BaseEstimator = RandomForestClassifier(), method: str = 'entropy', custom_method: Callable[[np.ndarray], np.ndarray] = None, n_groups: int = 3):
        if not hasattr(base_estimator, 'partial_fit'):
            raise ValueError("The base_estimator must implement the partial_fit method.")
        if method not in ['entropy', 'avg_confidence'] and custom_method is None:
            raise ValueError(
                "Invalid method argument. Should be 'entropy', 'avg_confidence', or a custom method must be provided.")
        if not isinstance(n_groups, int) or n_groups <= 0:
            raise ValueError("n_groups must be a positive integer.")

        self.base_estimator = base_estimator
        self.method = method
        self.custom_method = custom_method
        self.n_groups = n_groups

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """
        Fit the model according to the given training data.

        Parameters:
        -----------
        X : pd.DataFrame
            The training input samples.
        y : np.ndarray
            The target values (class labels) as integers or strings.

        Returns:
        --------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y)

        groups = group_by_difficulty(X, y, method=self.method, custom_method=self.custom_method, n_groups=self.n_groups)

        for X_group, y_group in groups:
            self.base_estimator.partial_fit(X_group, y_group, classes=unique_labels(y))

        return self

    def predict(self, X: pd.DataFrame):
        """
        Perform classification on samples in X.

        Parameters:
        -----------
        X : pd.DataFrame
            The input samples.

        Returns:
        --------
        y_pred : np.ndarray
            The predicted classes.
        """
        check_is_fitted(self)

        X = check_array(X)

        return self.base_estimator.predict(X)