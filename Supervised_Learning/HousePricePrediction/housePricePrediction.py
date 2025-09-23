import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df=pd.read_csv('data.csv')
#print(df.info())
#print(df.columns)


corr=df['Size'].corr(df['Price'])
#print(corr)


plt.scatter(df['Size'],df['Price'])
plt.xlabel('Size')
plt.ylabel('Price')
plt.title('Size vs Price')
#plt.show()

#5000,2300000

x=df.drop('Price',axis=1)
y=df['Price']

poly =PolynomialFeatures(degree=4)
sq=poly.fit_transform(x)

model =LinearRegression()
model.fit(sq,y)

predicted_value=model.predict(poly.fit_transform([[5000]]))
print(predicted_value)


