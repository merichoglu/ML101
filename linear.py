import numpy as np
import sklearn.metrics
import joblib


class LinearRegression:

    def __init__(
        self,
        method="normal_equation",
        learning_rate=0.01,
        epochs=1000,
        regularization=None,
        alpha=0.01,
    ):
        self.coefficients = None
        self.intercept = None
        self.method = method
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.alpha = alpha  # Regularization strength

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        if self.method == "normal_equation":
            self._fit_normal_equation(X, y)

        elif self.method == "gradient_descent":
            self._fit_gradient_descent(X, y)

        else:
            raise ValueError("Method must be 'normal_equation' or 'gradient_descent'")

    def _fit_normal_equation(self, X, y):
        """
        Fit model according to formula Y = w*X + b where w is weigh and b is bias
        """
        X_b = np.c_[(np.ones(X.shape[0], 1)), X]  # X_b is the bias term
        if self.regularization == "ridge":
            L = self.alpha * np.eye(X_b.shape[1])
            L[0, 0] = 0  # do not regularize the intercept term
            theta_best = np.linalg.inv(X_b.T.dot(X_b) + L).dot(X_b.T).dot(y)
        else:
            theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = theta_best[0]
        self.coefficients = theta_best[1:]

    def _fit_gradient_descent(self, X, y):
        """
        Fit model using gradient descent
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # X_b is the bias term
        m = len(y)
        theta = np.random.randn(X_b.shape[1])  # random initialization of theta

        for epoch in range(self.epochs):
            gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
            if self.regularization == "lasso":
                gradients += self.alpha * np.sign(theta)
            elif self.regularization == "ridge":
                gradients += self.alpha * theta
                gradients[0] -= self.alpha * theta[0]   # do not regularize the intercept term
            theta -= self.learning_rate * gradients

        self.intercept = theta[0]
        self.coefficients = theta[1:]

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.intercept, self.coefficients])

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        total_variance = np.var(y_true) * len(y_true)
        explained_variance = total_variance - np.sum((y_true - y_pred) ** 2)
        return explained_variance / total_variance

    def save_model(self, filepath):
        joblib.dump(self, filepath)

    def load_model(self, filepath):
        return joblib.load(filepath)
