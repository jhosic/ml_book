import numpy as np


class Perceptron(object):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, training_features, targets):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self.weights_ = np.zeros(1 + training_features.shape[1])  #a weight for every features
        self.errors_ = []

        for _ in range(self.n_iter):
            errors_count = 0
            for xi, target in zip(training_features, targets):
                update = self.eta * (target - self.predict(xi))
                self.weights_[1:] += update * xi
                self.weights_[0] += update
                errors_count += int(update != 0.0)
            self.errors_.append(errors_count)
        return self

    def net_input(self, training_features):
        """Calculate net input"""
        return np.dot(training_features, self.weights_[1:]) + self.weights_[0]

    def predict(self, training_features):
        """Return class label after unit step"""
        return np.where(self.net_input(training_features) >= 0.0, 1, -1)
