# noinspection PyShadowingNames

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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
        y = np.array(y).reshape(-1, 1)  # Ensure y is a column vector

        if self.method == "normal_equation":
            self._fit_normal_equation(X, y)
        elif self.method == "gradient_descent":
            self._fit_gradient_descent(X, y)
        else:
            raise ValueError("Method must be 'normal_equation' or 'gradient_descent'")

    def _fit_normal_equation(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        if self.regularization == "ridge":
            L = self.alpha * np.eye(X_b.shape[1])
            L[0, 0] = 0  # Do not regularize the intercept term
            theta_best = np.linalg.inv(X_b.T.dot(X_b) + L).dot(X_b.T).dot(y)
        else:
            theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept = theta_best[0, 0]
        self.coefficients = theta_best[1:].flatten()

    def _fit_gradient_descent(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        m = len(y)
        theta = np.random.randn(X_b.shape[1], 1)

        for epoch in range(self.epochs):
            gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
            if self.regularization == "ridge":
                regularization_term = self.alpha * np.r_[np.zeros((1, 1)), theta[1:]]
                gradients += 2 * regularization_term
            theta -= self.learning_rate * gradients

        self.intercept = theta[0, 0]
        self.coefficients = theta[1:].flatten()

    def predict(self, X):
        """
        Calculate Y given the X value using the model.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(np.r_[self.intercept, self.coefficients])

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """
        Calculate mean squared error according to the formula.
        """
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def r2_score(y_true, y_pred):
        """
        Determine the proportion of variance in Y that can be explained by X.
        In other words, a measure of how well the data fits our model.
        """
        total_variance = np.sum((y_true - np.mean(y_true)) ** 2)
        explained_variance = np.sum((y_true - y_pred) ** 2)
        return 1 - explained_variance / total_variance

    """
    Following two functions are for saving the model for later use.
    If you want to test your model on other datasets, you can save the model to your local.
    """

    def save_model(self, filepath):
        joblib.dump(self, filepath)

    @staticmethod
    def load_model(filepath):
        return joblib.load(filepath)


california_housing = fetch_california_housing(as_frame=True)
california_housing = california_housing

print(california_housing.DESCR)  # see the official description of our dataset

california_housing = california_housing.frame

california_housing.hist(bins=50, figsize=(12, 8))
plt.show()

california_housing.plot(
    kind="scatter",
    x="Longitude",
    y="Latitude",
    c="MedHouseVal",
    cmap="jet",
    colorbar=True,
    legend=True,
    sharex=False,
    figsize=(10, 7),
    s=california_housing["Population"] / 100,
    label="population",
    alpha=0.7,
)
plt.show()

corr = california_housing.corr()
corr["MedHouseVal"].sort_values(ascending=True)

print("Check for NA / NaN values")
print("------------------------")
print(california_housing.isna().sum())
print("------------------------")
print("Check the datatype of each column")
print(california_housing.dtypes)

scale = StandardScaler()

X = california_housing.drop("MedHouseVal", axis=1)  # dropping target column
y = california_housing["MedHouseVal"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = scale.fit_transform(X_train)
X_test = scale.transform(X_test)

model = LinearRegression(
    method="gradient_descent",
    learning_rate=0.01,
    epochs=2000,
    regularization="ridge",
    alpha=0.01,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse = model.mean_squared_error(y_test, predictions)
r2 = model.r2_score(y_test, predictions)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs Predicted Prices")
plt.show()
