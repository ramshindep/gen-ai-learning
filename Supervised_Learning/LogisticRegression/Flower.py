import pandas as pd
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('iris.csv')

encoder = LabelEncoder()

x = df.drop('species',axis =1)
y = df['species']

y = encoder.fit_transform(y)

model = LogisticRegression()

model.fit(x,y)

output = model.predict(pd.DataFrame([[6.6,2.9,4.6,1.3]],columns=['sepal_length','sepal_width','petal_length','petal_width']))

predicted_species = encoder.inverse_transform(output)

print(predicted_species)