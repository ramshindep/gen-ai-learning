import pandas as pd
from sklearn.linear_model import LinearRegression

df=pd.read_csv('salary_data.csv')
df.info()

new_df=df.dropna()

#print(new_df)

x=new_df.drop('Salary',axis=1)
y=new_df['Salary']

model =LinearRegression()

model.fit(x,y)

# salaries=model.predict(pd.DataFrame([2,6],columns=['Experience']))
# print(salaries[0])
# print(salaries[1])

missing_exp = df[df['Salary'].isna()][['Experience']]
salaries = model.predict(missing_exp)
print(salaries)

filled_df = df.copy()
filled_df.loc[filled_df['Salary'].isna(), 'Salary'] = salaries


print(filled_df)

a=filled_df.drop('Salary',axis=1)
b=filled_df['Salary']

model.fit(a,b)



salaries=model.predict(pd.DataFrame([16],columns=['Experience']))
print(salaries)

