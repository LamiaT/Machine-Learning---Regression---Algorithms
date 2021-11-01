"""Polynomial Regression for Machine Learning."""

# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv("dataset.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Training the Polynomial Regression model on the whole dataset
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = "blue")
plt.plot(X, lin_reg.predict(X), color = "green")
plt.title("Linear Regression")
plt.xlabel(" ")
plt.ylabel(" ")
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = "blue")
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = "green")
plt.title("Polynomial Regression")
plt.xlabel(" ")
plt.ylabel(" ")
plt.show()

# Visualising the Polynomial Regression results (with a higher resolution and a smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = "blue")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "green")
plt.title("Polynomial Regression with higher resolution")
plt.xlabel(" ")
plt.ylabel(" ")
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
