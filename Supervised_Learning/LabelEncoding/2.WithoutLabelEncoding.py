import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

df =pd.read_csv('salary_data_transformed.csv')

x=df.drop('Salary',axis=1)
y=df['Salary']

encoder =LabelEncoder()
x['Title']=encoder.fit_transform(x['Title'])
#print(x)

model =LinearRegression()
model.fit(x,y)

output =model.predict([[1,8]])
print(output)
