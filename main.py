import sys

import numpy
import pandas as pd
from fairlearn.datasets import fetch_adult

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import seaborn as sns
import matplotlib.pyplot as plt

from curriculum_classifier.utils.utils import get_learning_order

if __name__ == '__main__':
    df: pd.DataFrame = fetch_adult(as_frame=True)['data']

    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.drop(columns=['race'], axis=1).select_dtypes(include=['object', 'category']).columns

    numeric_imputer = SimpleImputer(strategy='median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])
    df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])

    X: pd.DataFrame = df.drop(columns=['race'], axis=1)
    y: pd.Series = df['race']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    X_transformed = pipeline.fit_transform(X)

    probabilities = get_learning_order(X_transformed, y)

    print(probabilities)