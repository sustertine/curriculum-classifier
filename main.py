import pandas as pd
from fairlearn.datasets import fetch_adult

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.classifiers.curriculum_classifier import CurriculumClassifier
from src.utils.utils import group_by_difficulty

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

    X_transformed = pipeline.fit_transform(X).toarray()

    split = group_by_difficulty(X_transformed, y, method='avg_confidence', n_groups=3)
    csf = CurriculumClassifier()