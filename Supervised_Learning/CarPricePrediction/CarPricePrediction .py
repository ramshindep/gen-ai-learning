import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df=pd.read_csv('car_price_data.csv')
print(df.columns)

df.info()

crr_age_price =df['Age'].corr(df['Price'])
crr_Milage_price=df['Mileage'].corr(df['Price'])

print(crr_age_price)
print(crr_Milage_price)

plt.scatter(df['Age'],df['Price'])
plt.xlabel('Age')
plt.ylabel('Price')
plt.title("Age vs price")
#plt.show()

model = LinearRegression()

x= df.drop('Price', axis=1)
y=df['Price']

model.fit(x,y)

predicted_price=model.predict(pd.DataFrame([[5,50000]],columns=['Age','Mileage']))

print(f"Predicted price is :{predicted_price}")
