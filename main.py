import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("KEEN_Yearly.csv")
features = ["Open", "High", "Low", "Volume", "Adj Close"]

X = df[features].iloc[:-1, :].dropna().values
y = df["Close"].iloc[1:].dropna().values

# 1 ) Multiple Linear Regression

# regressor = LinearRegression()
# regressor.fit(X, y)
# y_predict = regressor.predict(X)

# np.set_printoptions(precision=2)
# print(np.concatenate((y_predict.reshape(len(y_predict), 1), y.reshape(len(y), 1)), 1))

# plt.scatter(np.linspace(1, 20, num=20), y[-20:], color="red")
# plt.scatter(np.linspace(1, 20, num=20), y_predict[-20:], color="blue")

# 2 ) SVR

# y = y.reshape(len(y), 1)
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X = sc_X.fit_transform(X)
# y = sc_y.fit_transform(y)

# regressor = SVR(kernel="rbf")
# regressor.fit(X, y)

# y_predict = regressor.predict(X)

# np.set_printoptions(precision=2)
# print(np.concatenate((sc_y.inverse_transform(y_predict.reshape(-1, 1)), sc_y.inverse_transform(y)), 1))

# plt.scatter(np.linspace(1, 20, num=20), sc_y.inverse_transform(y).flatten()[-20:], color="red")
# plt.scatter(np.linspace(1, 20, num=20), sc_y.inverse_transform(y_predict.reshape(-1, 1))[-20:], color="blue")

# print(regressor.predict([[805.00, 815.00, 785.00, 9058400, 805]]))

# 3 Decision Tree

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

y_predict = regressor.predict(X_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_predict.reshape(len(y_predict), 1), y_test.reshape(len(y_test), 1)), 1))

plt.scatter(np.linspace(1, 30, num=30), y_test[-30:], color="red")
plt.scatter(np.linspace(1, 30, num=30), y_predict[-30:], color="blue")

# Predict future price (Args = Open, High, Low, Volume, Adj Close)
print(regressor.predict([[805, 815, 785, 9058400, 805]]))