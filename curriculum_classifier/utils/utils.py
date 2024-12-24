from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def get_sensitive_groups(X: pd.DataFrame, y: np.array) -> List[float]:
    """
       Get the sensitive groups from the dataframe
    """
    random_forest_classifier: RandomForestClassifier = RandomForestClassifier(n_estimators=100, random_state=123)
    random_forest_classifier.fit(X, y)
    probabilities: List[float] = random_forest_classifier.predict_proba(X)
    return probabilities
