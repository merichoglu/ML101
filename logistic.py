import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
titanic = pd.read_csv(url)

# See the column descriptions for dataset
titanic.info()

# Column descriptions
column_descriptions = {
    'PassengerId': 'A unique ID for each passenger',
    'Survived': 'Survival (0 = No, 1 = Yes)',
    'Pclass': 'Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)',
    'Name': 'Name of the passenger',
    'Sex': 'Sex of the passenger',
    'Age': 'Age of the passenger',
    'SibSp': 'Number of siblings/spouses aboard the Titanic',
    'Parch': 'Number of parents/children aboard the Titanic',
    'Ticket': 'Ticket number',
    'Fare': 'Fare paid by the passenger',
    'Cabin': 'Cabin number',
    'Embarked': 'Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)'
}

print("\nColumn Descriptions:")
for column, description in column_descriptions.items():
    print(f"{column}: {description}")

missing_values = titanic.isnull().sum()
print("\nCount of missing values in each column:")
print(missing_values)

# Visualize missing values using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(titanic.isnull(), cbar=False, cmap='viridis')
plt.title('Heatmap of Missing Values in Titanic Dataset')
plt.show()

# See the first few rows of the dataset
titanic.head()

# Now let's clear the dataset by handling the missing values.

# Fill missing values for 'Age' with the median value
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

# Fill missing values for 'Embarked' with the most frequent value
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])

# Drop 'Cabin' if it exists, as it has many missing values
if 'Cabin' in titanic.columns:
    titanic = titanic.drop('Cabin', axis=1)

# Encode categorical variables
label_enc = LabelEncoder()
titanic['Sex'] = label_enc.fit_transform(titanic['Sex'].astype(str))
titanic['Embarked'] = label_enc.fit_transform(titanic['Embarked'].astype(str))

# Verify the encoding
unique_values = titanic['Sex'].unique()
print(f"Unique values in 'Sex' column: {unique_values}")

# Now let's see what the dataset looks like!

# Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(titanic['Age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Count of Survived and Not Survived
plt.figure(figsize=(10, 6))
sns.countplot(x='Survived', data=titanic)
plt.title('Count of Survived and Not Survived')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Survival Rate by Passenger Class
plt.figure(figsize=(10, 6))
sns.barplot(x='Pclass', y='Survived', data=titanic)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Survival Rate by Gender
plt.figure(figsize=(10, 6))
sns.barplot(x='Sex', y='Survived', data=titanic)
plt.title('Survival Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.show()

# Survival Rate by Embarkation Point
plt.figure(figsize=(10, 6))
sns.barplot(x='Embarked', y='Survived', data=titanic)
plt.title('Survival Rate by Embarkation Point')
plt.xlabel('Embarked')
plt.ylabel('Survival Rate')
plt.show()

# Heatmap of Feature Correlations with Survival
plt.figure(figsize=(10, 8))
# Dropping non-numeric columns
numeric_titanic = titanic.drop(columns=['Name', 'Ticket', 'PassengerId'])
corr_matrix = numeric_titanic.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap of Feature Correlations with Survival')
plt.show()

# Select features and target variable
X = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = titanic['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, regularization=None, alpha=0.01):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.alpha = alpha
        self.coefficients = None
        self.intercept = None

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        m, n = X_b.shape
        theta = np.random.randn(n, 1)

        for epoch in range(self.epochs):
            predictions = self._sigmoid(X_b.dot(theta))
            errors = predictions - y
            gradients = X_b.T.dot(errors) / m
            if self.regularization == "ridge":
                regularization_term = self.alpha * np.r_[np.zeros((1, 1)), theta[1:]]
                gradients += regularization_term / m
            theta -= self.learning_rate * gradients

        self.intercept = theta[0, 0]
        self.coefficients = theta[1:].flatten()

    def predict_proba(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return self._sigmoid(X_b.dot(np.r_[self.intercept, self.coefficients]))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)


# Now let's train the model on our dataset:
model = LogisticRegression(learning_rate=0.01, epochs=10000, regularization="ridge", alpha=0.01)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
proba_predictions = model.predict_proba(X_test)

accuracy = model.accuracy(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Approximately 80%, not bad at all! Now let's analyze the performance even further.

conf_matrix = confusion_matrix(y_test, predictions)
class_report = classification_report(y_test, predictions)

print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

fpr, tpr, _ = roc_curve(y_test, proba_predictions)
roc_auc_scratch = roc_auc_score(y_test, proba_predictions)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_scratch:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Scratch)')
plt.legend(loc="lower right")
plt.show()

arbitrary_passenger = pd.DataFrame({
    'Pclass': [3],
    'Sex': [0],
    'Age': [25],
    'SibSp': [1],
    'Parch': [0],
    'Fare': [10],
    'Embarked': [2]
})

# Scale the parameters for this arbitrary passenger, make sure to use the same scaler fitted on the training data
arbitrary_passenger_scaled = scaler.transform(arbitrary_passenger)

# Make a prediction
predicted_proba = model.predict_proba(arbitrary_passenger_scaled)
predicted_survival = model.predict(arbitrary_passenger_scaled)

print(f"Predicted survival probability: {predicted_proba[0]}")
print(f"Predicted survival (0 = Not Survived, 1 = Survived): {predicted_survival[0]}")

# See the passenger on the sigmoid graph

sigmoid_value = model.predict_proba(arbitrary_passenger_scaled)
z = np.linspace(-10, 10, 200)
sigmoid = 1 / (1 + np.exp(-z))

plt.figure(figsize=(10, 6))
plt.plot(z, sigmoid, label='Sigmoid Function')
plt.scatter([sigmoid_value], [1 / (1 + np.exp(-sigmoid_value))], color='red', zorder=5, label='Arbitrary Passenger')
plt.axhline(0.5, color='gray', linestyle='--', label='Decision Boundary')
plt.title('Sigmoid Function with Arbitrary Passenger')
plt.xlabel('z')
plt.ylabel('Sigmoid(z)')
plt.legend()
plt.show()
