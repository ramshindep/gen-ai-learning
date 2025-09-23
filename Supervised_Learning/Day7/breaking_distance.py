import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df =pd.read_csv('data.csv')
#print(df.info())

correlation=df['Speed'].corr(df['BrakingDistance'])

#print(correlation)

plt.scatter(df['Speed'],df['BrakingDistance'])
plt.xlabel('Speed')
plt.ylabel('BrakingDistance')
plt.title('Speed vs BreakingDistance')
#plt.show()

x =df.drop('BrakingDistance', axis=1)
y=df['BrakingDistance']


poly =PolynomialFeatures(degree=6)
cube=poly.fit_transform(x)

model =LinearRegression()
model.fit(cube,y)

predicted_value=model.predict(poly.fit_transform([[120]]))
print(predicted_value)