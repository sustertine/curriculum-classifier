# Curriculum Classifier

## Overview

This project implements a curriculum classifier that groups data by learning difficulty and trains a classifier on these groups. The classifier is evaluated on both predictive performance and fairness metrics.

## Project Structure

- `src/classifiers/curriculum_classifier.py`: Contains the implementation of the `CurriculumClassifier`.
- `src/utils/learning_difficulty.py`: Contains utility functions for grouping data by learning difficulty.
- `main.py`: Script to preprocess data, train the classifier, and evaluate its performance.
- `notebooks/demo.ipynb`: Jupyter notebook demonstrating the usage of the curriculum classifier and comparing its performance with a standard classifier.
- `requirements.txt`: Lists the dependencies required for the project.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/sustertine/group-sensitive-curriculum-classifier.git
    cd group-sensitive-curriculum-classifier
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
   
### Curriculum Classifier

The `CurriculumClassifier` is a custom classifier that groups data by learning difficulty and trains a base estimator on these groups.

#### Parameters

- `base_estimator` \(`BaseEstimator`\): The base estimator to be used for training. Must implement the `partial_fit` method.
- `method` \(`str`, default='entropy'\): The method to calculate difficulty \('entropy' or 'avg_confidence'\).
- `custom_method` \(`Callable[[np.ndarray], np.ndarray]`, optional\): A custom method to calculate difficulty. If provided, it overrides the native methods.
- `n_groups` \(`int`, default=3\): The number of groups to split the data into.

#### Methods

- `fit(X, y)`: Fits the classifier to the data.
- `predict(X)`: Predicts the classes for the input samples.

#### Example

```python
from sklearn.linear_model import SGDClassifier
from src.classifiers.curriculum_classifier import CurriculumClassifier

curriculum_classifier = CurriculumClassifier(base_estimator=SGDClassifier(), method='entropy', n_groups=3)
curriculum_classifier.fit(X_train, y_train)
y_pred = curriculum_classifier.predict(X_test)

```

### Difficulty Methods

#### Entropy
The entropy method calculates the difficulty of a sample based on the entropy of the predicted class probabilities. Entropy measures the uncertainty in the predictions, with higher entropy indicating more uncertainty and thus higher difficulty.

**Formula:**
\[ \text{Entropy} = -\sum p(x) \log(p(x)) \]
where \( p(x) \) is the predicted probability for class \( x \).

**Example:**
If a sample has predicted probabilities \([0.5, 0.5]\), the entropy is:
\[
- (0.5 \log(0.5) + 0.5 \log(0.5)) = 1.0
\]

#### Average Confidence
The average confidence method calculates the difficulty based on the average confidence of the classifier in its predictions. Lower confidence indicates higher difficulty.

**Steps:**
1. Train a simple classifier (e.g., RandomForest) on the data.
2. Calculate the average confidence for each class.
3. Assign difficulty scores based on the confidence of the classifier in its predictions for each sample.

**Example:**
If the average confidence for a class is \( 0.8 \) and the classifier's confidence for a sample is \( 0.6 \), the difficulty score is lower, indicating higher difficulty.
