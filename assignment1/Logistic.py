import numpy as np


class LogisticRegression:

    def __init__(self, penalty="l2", gamma=100, fit_intercept=True):
        """
        Parameters:
        - penalty: str, "l1" or "l2". Determines the regularization to be used.
        - gamma: float, regularization coefficient. Used in conjunction with 'penalty'.
        - fit_intercept: bool, whether to add an intercept (bias) term.
        """
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.coef_ = None

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        ################################################################################
        # TODO:                                                                        #
        # Implement the sigmoid function.
        ################################################################################
        return 1.0 / (1.0 + np.exp(-x))
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################

    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e3):
        """
        Fit the regression coefficients via gradient descent or other methods

        Parameters:
        - X: numpy array of shape (n_samples, n_features), input data.
        - y: numpy array of shape5 (n_samples,), target data.
        - lr: float, learning rate for gradient descent.
        - tol: float, tolerance to decide convergence of gradient descent.
        - max_iter: int, maximum number of iterations for gradient descent.
        Returns:
        - losses: list, a list of loss values at each iteration.
        """
        # If fit_intercept is True, add an intercept column
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Initialize coefficients
        self.coef_ = np.zeros(X.shape[1])

        # List to store loss values at each iteration
        losses = []

        ################################################################################
        # TODO:                                                                        #
        # Implement gradient descent with optional regularization.
        # 1. Compute the gradient
        # 2. Apply the update rule
        # 3. Check for convergence
        ################################################################################
        for i in range(int(max_iter)):
            # Compute the linear combination of inputs and weights
            linear_output = np.dot(X, self.coef_)
            # Apply the sigmoid function to the linear combination
            y_pred = self.sigmoid(linear_output)
            epsilon = 1e-15
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            # Compute the gradient
            if self.penalty == "l1":
                gradient = -np.dot(X.T, (y - y_pred))/X.shape[0] + self.gamma * np.sign(self.coef_) / X.shape[0]
            elif self.penalty == "l2":
                gradient = -np.dot(X.T, (y - y_pred))/X.shape[0] + self.gamma * self.coef_ / X.shape[0]
            # Update the weights
            self.coef_ -= lr * gradient
            # Compute the loss function
            loss = -np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            if self.penalty == "l1":
                loss += self.gamma * np.sum(np.abs(self.coef_))/X.shape[0]
            elif self.penalty == "l2":
                loss += (self.gamma/2) * np.sum(self.coef_**2)/X.shape[0]
            # Check for convergence
            losses.append(loss)
            if np.linalg.norm(gradient) < tol:
                break
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
        return losses

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.

        Parameters:
        - X: numpy array of shape (n_samples, n_features), input data.

        Returns:
        - probs: numpy array of shape (n_samples,), prediction probabilities.
        """
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # Compute the linear combination of inputs and weights
        linear_output = np.dot(X, self.coef_)

        ################################################################################
        # TODO:                                                                        #
        # Task3: Apply the sigmoid function to compute prediction probabilities.
        ################################################################################
        probs = self.sigmoid(linear_output)
        #化为分类变量
        probs = np.where(probs > 0.5, 1, 0)
        return probs
        ################################################################################
        #                                 END OF YOUR CODE                             #
        ################################################################################
