import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils.multiclass import unique_labels
from typing import Callable, List, Tuple
import numpy as np
import pandas as pd
from src.utils.learning_difficulty import group_by_difficulty

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(filename)s:%(lineno)s - %(funcName)20s() - [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

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

    def __init__(self, base_estimator: BaseEstimator, method: str = 'entropy', custom_method: Callable[[np.ndarray], np.ndarray] = None, n_groups: int = 3):
        if not hasattr(base_estimator, 'partial_fit'):
            logger.error(f"The base_estimator must implement the partial_fit method. {hasattr(base_estimator, 'partial_fit')}")
            raise ValueError("The base_estimator must implement the partial_fit method.")
        if method not in ['entropy', 'avg_confidence'] and custom_method is None:
            raise ValueError("Invalid method argument. Should be 'entropy', 'avg_confidence', or a custom method must be provided.")
        if not isinstance(n_groups, int) or n_groups <= 0:
            raise ValueError("n_groups must be a positive integer.")

        logger.debug(f"partial_fit present: {hasattr(base_estimator, 'partial_fit')}")

        self.base_estimator = base_estimator
        self.method = method
        self.custom_method = custom_method
        self.n_groups = n_groups
        self._is_fitted = False

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
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Split the dataset into groups based on difficulty
        groups = group_by_difficulty(X, y, method=self.method, custom_method=self.custom_method, n_groups=self.n_groups)

        # Fit the base estimator on each group using partial_fit
        for X_group, y_group in groups:
            logger.debug(f"Fitting group with {len(X_group)} samples")
            self.base_estimator.partial_fit(X_group, y_group, classes=unique_labels(y))

        self._is_fitted = True
        if hasattr(self.base_estimator, "_sklearn_is_fitted_"):
            self.base_estimator._sklearn_is_fitted_ = True
        logger.debug("Fitting completed")
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
        # Check if fit has been called
        check_is_fitted(self, '_is_fitted')

        # Input validation
        X = check_array(X)

        return self.base_estimator.predict(X)