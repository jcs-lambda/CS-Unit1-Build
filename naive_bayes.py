"""Gaussian Naive Bayes Classifier
"""


import math

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted


class NaiveBayes(BaseEstimator, ClassifierMixin):
    """Gaussian Naive Bayes Classifier"""

    def __init__(self):
        """No initialization parameters."""
        pass

    def _validate_input(self, X, y=None):
        """Returns validated input.

        :param X: 2d array-like of numeric values with no NaNs or infinite values

        :param y: 1d array-like of hashable values with no NaNs or infinite values

        :return: validated data, converted to numpy arrays
        """
        if y is not None:
            # fitting the model, validate X and y
            return check_X_y(X, y)
        else:
            # predicting, validate X
            check_is_fitted(self, ['num_features_', 'feature_summaries_'])
            X = check_array(X)
            if X.shape[1] != self.num_features_:
                raise(ValueError('unexpected input shape: (x, {X.shape[1]}); must be (x, {self.num_features_})'))
            return X

    def fit(self, X, y):
        """Fit the model with training data. X and y must be of equal length.

        :param X: 2d array-like of numeric values with no NaNs or infinite values

        :param y: 1d array-like of hashable values with no NaNs or infinite values
        
        :return: fitted instance
        """
        X, y = self._validate_input(X, y)
        self.num_features_ = X.shape[1]

        # create dictionary containing input data separated by class label
        data_by_class = {}
        for i in range(len(X)):
            features = X[i]
            label = y[i]
            if label not in data_by_class:
                # first occurence of label, create empty list in dictionary
                data_by_class[label] = []
            data_by_class[label].append(features)
        
        # summarize the distribution of features by label as list of
        # (mean, standard deviation) tuples
        # store in instance attribute for use in prediction
        self.feature_summaries_ = {}
        for label, features in data_by_class.items():
            self.feature_summaries_[label] = [
                (np.mean(column), np.std(column))
                for column in zip(*features)
            ]

        return self

    def _liklihood(self, x, mean, stdev):
        """Calculate conditional probability of a Gaussian distribution.

        :param x: float
        
        :param mean: float, sample mean

        :param stdev: float, sample standard deviation

        :return: float
        """
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
    
    def predict(self, X):
        """Returns class predictions for each row in X.

        :param X: 2d array-like of numeric values with no NaNs or infinite values
        whose .shape[1] == .shape[1] of fitted data

        :return: np.array of class predictions
        """
        X = self._validate_input(X)

        # predicted class labels
        predictions = []

        # iterate input rows
        for x in X:
            # get cumulative log probabilites for each class for this row
            probabilities = {}
            for label, features in self.feature_summaries_.items():
                probabilities[label] = 0
                for i in range(len(features)):
                    mean, stdev = features[i]
                    probabilities[label] += math.log2(
                        self._liklihood(x[i], mean, stdev)
                    )

            # find class with highest probability
            best_label, best_prob = None, -1
            for label, probability in probabilities.items():
                if best_label is None or probability > best_prob:
                    best_prob = probability
                    best_label = label

            # prediction for this row
            predictions.append(best_label)

        return np.array(predictions)
