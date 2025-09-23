import pandas as pd
from sklearn.impute import KNNImputer

df =pd.read_csv('salary_data.csv')

imputer =KNNImputer(n_neighbors=3)

df['Salary']=imputer.fit_transform(df[['Salary']])

print(df)

