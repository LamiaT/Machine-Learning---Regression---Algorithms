"""Support Vector Machine Regression (SVMR) for Machine Learning."""

# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Importing the dataset
dataset = pd.read_csv("dataset.csv")
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(len(y), 1)

# Feature Scaling
sc_X = StandardScaler()
sc_y = StandardScaler()

X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Training the SVR model on the whole dataset
regressor = SVR(kernel = "rbf")
regressor.fit(X, y)

# Predicting a new result
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X),
            sc_y.inverse_transform(y),
            color = "blue")

plt.plot(sc_X.inverse_transform(X),
         sc_y.inverse_transform(regressor.predict(X)),
         color = "green")

plt.title("SVMR")
plt.xlabel(" ")
plt.ylabel(" ")
plt.show()

# Visualising the SVR results (with Higher Resolution)
X_grid = np.arange(min(sc_X.inverse_transform(X)),
                   max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(sc_X.inverse_transform(X),
            sc_y.inverse_transform(y),
            color = "blue")

plt.plot(X_grid,
         sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))),
         color = "green")

plt.title("SVMR with higher resolution")
plt.xlabel(" ")
plt.ylabel(" ")
plt.show()
