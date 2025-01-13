import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from typing import Callable, List, Tuple
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s]: %(message)s')


def group_by_difficulty(
        X: pd.DataFrame,
        y: np.ndarray,
        method: str = 'entropy',
        custom_method: Callable[[np.ndarray], np.ndarray] = None,
        n_groups: int = 3
) -> List[Tuple[pd.DataFrame, np.ndarray]]:
    """
    Split the dataset into n groups based on learning difficulty.

    :param X: Features
    :param y: Target
    :param method: Method to calculate difficulty ('entropy' or 'avg_confidence').
    :param custom_method: Custom method to calculate difficulty. If provided, it overrides the native methods.
    :param n_groups: Number of groups to split the data into.
    :return: List of tuples containing (X_group, y_group) for each group.
    """
    if method not in ['entropy', 'avg_confidence']:
        raise ValueError("Invalid method argument. Should be 'entropy' or 'avg_confidence'.")

    if custom_method is not None and not callable(custom_method):
        raise ValueError("The custom_method must be a callable function.")

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    random_forest_classifier = RandomForestClassifier(n_estimators=1, random_state=123)
    random_forest_classifier.fit(X, y)
    probabilities = random_forest_classifier.predict_proba(X)

    difficulty_scores = np.ndarray([])

    if custom_method:
        logging.debug("Using custom method to calculate difficulty.")
        difficulty_scores = custom_method(probabilities)
    else:
        if method == 'entropy':
            logging.debug("Using entropy to calculate difficulty.")
            difficulty_scores = -np.sum(probabilities * np.log(probabilities + 1e-9), axis=1)
        elif method == 'avg_confidence':
            logging.debug("Using average class confidence to calculate difficulty.")
            avg_class_confidences = probabilities.mean(axis=0)
            class_to_index = {cls: idx for idx, cls in enumerate(np.unique(y))}
            difficulty_scores = np.array([avg_class_confidences[class_to_index[class_label]] for class_label in y])

    sorted_indices = np.argsort(difficulty_scores)
    groups = np.array_split(sorted_indices, n_groups)

    return [(X.iloc[group], y[group]) for group in groups]
