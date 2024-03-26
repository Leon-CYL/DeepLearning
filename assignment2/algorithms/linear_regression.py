"""
Linear Regression model
"""

import numpy as np


class Linear(object):
    def __init__(self, n_class: int, lr: float, epochs: int, weight_decay: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # Initialize in train
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.weight_decay = weight_decay

    def train(self, X_train: np.ndarray, y_train: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Train the classifier.

        Use the linear regression update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        N, D = X_train.shape
        self.w = weights

        # TODO: implement me
        
        # ONE-HOT ENCODING
        one_hot = np.full((N, self.n_class), -1, dtype=int)
        one_hot[np.arange(N), y_train] = 1
        
        #Training
        gradient = np.zeros((self.n_class, D))
        for j in range(self.epochs):
            loss = np.dot(X_train, self.w.T) - one_hot
            gradient += np.dot(loss.T, X_train) * 2 / N
            self.w = self.w - self.lr * (self.weight_decay * self.w + gradient)

        return self.w

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        N = X_test.shape[0]
        y_pred = np.zeros(N)
        for i in range(N):
            y_pred[i] = np.argmax(np.dot(self.w, X_test[i]), axis=0)
        return y_pred
