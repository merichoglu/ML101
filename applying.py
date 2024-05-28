import linear
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

housing = fetch_california_housing()
X = housing.data
y = housing.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

model = linear.LinearRegression(
    method="gradient_descent",
    learning_rate=0.1,
    epochs=10000,
    regularization="ridge",
    alpha=0.1,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mse = model.mean_squared_error(y_true=y_test, y_pred=predictions)
r2 = model.r2_score(y_true=y_test, y_pred=predictions)

print("Gradient Descent Method with Ridge Regularization:")
print("Coefficients:", model.coefficients)
print("Intercept:", model.intercept)
print("MSE:", mse)
print("R^2 Score:", r2)

# Random sampling for visualization
np.random.seed(42)
sample_indices = np.random.choice(len(y_test), size=500, replace=False)  # Random sample of 500 points

plt.figure(figsize=(10, 6))
plt.scatter(y_test[sample_indices], predictions[sample_indices], edgecolors=(0, 0, 0), alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values (Sampled)')
plt.legend()
plt.show()