import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.linear_model import LinearRegression

data_frame=pd.read_csv('salary_data.csv')

data_frame.info()

print(data_frame.describe)

plt.scatter(data_frame['Experience'],data_frame['Salary'])
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')
#plt.show() 

correlation =data_frame['Experience'].corr(data_frame['Salary'])
print(correlation)


covarience=np.cov(data_frame['Experience'],data_frame['Salary'])
print("Covarience: ",covarience)

print("Mean Salary: ",data_frame['Salary'].mean())
print("Median Salary: ",data_frame['Salary'].median())
print("Mode Salary: ",data_frame['Salary'].mode()[0])

x = data_frame.drop('Salary',axis=1)

y=data_frame['Salary']

model =LinearRegression()

model.fit(x,y)

salaries=model.predict(pd.DataFrame([15],columns=['Experience']))

print("Salary of 15 years of is: ",salaries[0])


