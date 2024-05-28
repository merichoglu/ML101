import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing

# Load California housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Convert to DataFrame for easier exploration
df = pd.DataFrame(X, columns=data.feature_names)
df["MedHouseVal"] = y

# Basic Statistics
print("Basic Statistics:\n")
print(df.describe())

# Correlation Matrix
print("\nCorrelation Matrix:\n")
correlation_matrix = df.corr()
print(correlation_matrix)

# Visualize Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Pairplot
plt.figure(figsize=(12, 10))
sns.pairplot(df, height=2.5)
plt.suptitle("Pairplot of Features", y=1.02)
plt.show()

# Distribution of Target Variable
plt.figure(figsize=(8, 6))
sns.histplot(df["MedHouseVal"], bins=30, kde=True)
plt.title("Distribution of Median House Value")
plt.xlabel("Median House Value")
plt.ylabel("Frequency")
plt.show()

# Scatter plots of key features against the target
key_features = ["MedInc", "AveRooms", "AveOccup", "HouseAge"]

plt.figure(figsize=(14, 10))
for i, feature in enumerate(key_features):
    plt.subplot(2, 2, i + 1)
    sns.scatterplot(x=df[feature], y=df["MedHouseVal"], alpha=0.5)
    plt.title(f"{feature} vs Median House Value")
    plt.xlabel(feature)
    plt.ylabel("Median House Value")
plt.tight_layout()
plt.show()
