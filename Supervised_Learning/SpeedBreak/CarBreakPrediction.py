import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

df= pd.read_csv('')

corr =df['Speed'].corr(df['BreakingDistance'])

x=df[['Speed']]
y=df['BreakingDistance']

degree=2
model=make_pipeline(PolynomialFeatures(degree),LinearRegression())

model.fit(x,y)

output=model.predict(pd.DataFrame([[120]],columns=['Speed']))
print("Breaking Distance at speed 120 is: ",output[0])
