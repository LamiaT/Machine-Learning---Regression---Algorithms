"""Simple Linear Regression for Machine Learning."""

# Importing the necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv("dataset.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 1/3, 
                                                    random_state = 0)

# Training the Simple Linear Regression model on the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = "blue")
plt.plot(X_train, regressor.predict(X_train), color = "green")
plt.title("Training set")
plt.xlabel(" ")
plt.ylabel(" ")
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = "blue")
plt.plot(X_train, regressor.predict(X_train), color = "green")
plt.title("Test set")
plt.xlabel(" ")
plt.ylabel(" ")
plt.show()
