import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df=pd.read_csv('data.csv')

corr=df['Speed'].corr(df['BrakingDistance'])

print(corr)

plt.scatter(df['Speed'],df['BrakingDistance'])
plt.xlabel('Speed')
plt.ylabel('BrakingDistance')
plt.title('Speed vs Breaking Distance')
#plt.show()

x=df.drop('BrakingDistance',axis=1)
y=df['BrakingDistance']

poly =PolynomialFeatures(degree=4)
sq=poly.fit_transform(x)

model =LinearRegression()
model.fit(sq,y)

predict=model.predict(poly.fit_transform([[120]]))
print(predict)





