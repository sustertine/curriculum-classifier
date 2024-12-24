from typing import List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_learning_order(X: pd.DataFrame, y: pd.Series) -> List[float]:
    """
    This function takes in a dataframe and a numpy array and returns a list of unique values in the dataframe.
    :param X: Features
    :param y: Target
    :return:
    """
    random_forest_classifier: RandomForestClassifier = RandomForestClassifier(n_estimators=1, random_state=123)
    random_forest_classifier.fit(X, y)

    probabilities = random_forest_classifier.predict_proba(X)

    class_labels = random_forest_classifier.classes_
    average_probabilities = probabilities.mean(axis=0)
    per_class_probabilities = dict(zip(class_labels, average_probabilities))
    per_class_probabilities = {key: float(value) for key, value in per_class_probabilities.items()}

    return per_class_probabilities
