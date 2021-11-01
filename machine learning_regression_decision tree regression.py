"""Decision Tree Regression for Machine Learning."""

# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Importing the dataset
dataset = pd.read_csv("dataset.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
regressor.predict([[6.5]])

# Visualising the Decision Tree Regression results (with higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = "blue")
plt.plot(X_grid, regressor.predict(X_grid), color = "green")
plt.title("Decision Tree Regression")
plt.xlabel(" ")
plt.ylabel(" ")
plt.show()
