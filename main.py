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
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("SMLE_Yearly.csv")
features = ["Open", "High", "Low", "Volume", "Adj Close"]

X = df[features].iloc[:-1, :].values
y = df["Close"].iloc[1:].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 1 ) Multiple Linear Regression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)

np.set_printoptions(precision=2)
print(np.concatenate((y_predict.reshape(len(y_predict), 1), y_test.reshape(len(y_test), 1)), 1))
print(r2_score(y_test, y_predict))

# 2 ) SVR

# y_train = y_train.reshape(len(y_train), 1)
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# y_train = sc_y.fit_transform(y_train)

# regressor = SVR(kernel="rbf")
# regressor.fit(X_train, y_train)

# y_predict = regressor.predict(sc_X.transform(X_test))

# np.set_printoptions(precision=2)
# print(np.concatenate((sc_y.inverse_transform(y_predict.reshape(-1, 1)), y_test.reshape(-1, 1)), 1))
# print(r2_score(y_test, sc_y.inverse_transform(y_predict.reshape(-1, 1)).flatten()))

# 3 Decision Tree

# regressor = DecisionTreeRegressor(random_state=0)
# regressor.fit(X_train, y_train)

# y_predict = regressor.predict(X_test)

# np.set_printoptions(precision=2)
# print(np.concatenate((y_predict.reshape(len(y_predict), 1), y_test.reshape(len(y_test), 1)), 1))
# print(r2_score(y_test, y_predict))

# 4 Polynomial Linear Regression

# poly_reg = PolynomialFeatures(degree=5)
# X_poly = poly_reg.fit_transform(X_train)

# regressor = LinearRegression()
# regressor.fit(X_poly, y_train)

# y_predict = regressor.predict(poly_reg.transform(X_test))

# np.set_printoptions(precision=2)
# print(np.concatenate((y_predict.reshape(-1, 1), y_test.reshape(-1, 1)), 1))
# print(r2_score(y_test, y_predict))

# 5 Random Forest Regression

# regressor = RandomForestRegressor(random_state=0, n_estimators=100)
# regressor.fit(X_train, y_train)

# y_predict = regressor.predict(X_test)

# np.set_printoptions(precision=2)
# print(np.concatenate((y_predict.reshape(-1, 1), y_test.reshape(-1, 1)), 1))
# print(r2_score(y_test, y_predict))

# Predict future price here (Args = Open, High, Low, Volume, Adj Close)

print(regressor.predict([[132, 144, 132, 389982900, 143]]))
# print(sc_y.inverse_transform(regressor.predict(sc_X.transform([[2740, 2800, 2720, 12587100, 2740]])).reshape(-1, 1)))