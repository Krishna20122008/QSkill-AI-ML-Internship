import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

print("--- Running Task 1: Iris Flower Classification ---")

# 1. Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a simple K-Nearest Neighbors classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 4. Evaluate with accuracy and a classification report matrix
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))