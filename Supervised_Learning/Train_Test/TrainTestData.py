import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data=pd.read_csv()

x=data.drop("Spendings",axis=1)
y=data["Spendings"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1000)


model=LinearRegression()
model.fit(X_train,y_train)

train_score=model.score(X_train,y_train)

test_score=model.score(X_test,y_test)

print(f"Train Score:{train_score}")

print(f"Test Score:{test_score}")