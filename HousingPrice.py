from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df_test = pd.read_csv('D:/Datasets/HousingPrice/test.csv')
df_train = pd.read_csv('D:/Datasets/HousingPrice/train.csv')


x_train = df_train['GrLivArea'].values.reshape(-1,1)
y_train = df_train['SalePrice'].values.reshape(-1,1)
x_test = df_test['GrLivArea'].values.reshape(-1,1)

r = LinearRegression()
r.fit(x_train,y_train)
y_pred = r.predict(x_test)
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.plot(x_test,y_pred)
plt.show()
