import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv('Advertising.csv')
print(df.columns)
#print(df.info)

corr_tv_sales=df['TV'].corr(df['sales'])

print(f"coorelation of tv with sales is:{corr_tv_sales}")

corr_radio_sales=df['radio'].corr(df['sales'])
print(f"corelation of radio and sales: {corr_radio_sales}")

corr_newspaper_sales=df['newspaper'].corr(df['sales'])
print(f"corelation of newsPaper and Sales:{corr_newspaper_sales}")

plt.scatter(df['TV'],df['sales'])
plt.xlabel('TV')
plt.ylabel('sales')
plt.title("TV vs Sales")
#plt.show()

plt.scatter(df['newspaper'],df['sales'])
plt.xlabel('newspaper')
plt.ylabel('sales')
plt.title("NewsPaper vs Sales")
#plt.show()

plt.scatter(df['radio'],df['sales'])
plt.xlabel('radio')
plt.ylabel('sales')
plt.title("radio vs sales")
#plt.show()

model = LinearRegression()

x= df.drop('sales', axis=1)
y=df['sales']

model.fit(x,y)

predicted_sales=model.predict(pd.DataFrame([[]],columns=['TV','radio','newspaper']))
print(predicted_sales)