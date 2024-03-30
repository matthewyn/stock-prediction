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

df = pd.read_csv("GJTL_Yearly.csv")
features = ["Open", "High", "Low", "Volume", "Adj Close"]

X = df[features].iloc[:-1, :].dropna().values
y = df["Close"].iloc[1:].dropna().values

# 1 ) Multiple Linear Regression

regressor = LinearRegression()
regressor.fit(X, y)
y_predict = regressor.predict(X)

print(np.concatenate((y_predict.reshape(len(y_predict), 1), y.reshape(len(y), 1)), 1))

plt.scatter(np.linspace(1, 20, num=20), y[-20:], color="red")
plt.scatter(np.linspace(1, 20, num=20), y_predict[-20:], color="blue")

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