import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("--- Running Task 3: House Price Prediction ---")

# 1. Load the dataset (California housing prices)
housing = fetch_california_housing(as_frame=True)
X = housing.data   # Features: median income, house age, number of rooms, etc.
y = housing.target # Target: House price in units of $100,000

# 2. Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Make predictions and evaluate performance
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared Score (Accuracy): {r2:.2f}")