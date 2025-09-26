import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,f1_score


df=pd.read_csv('hearing_data.csv')

print(df['age'].corr(df['test_result']))
print(df['physical_score'].corr(df['test_result']))


hearing_problem=df[df['test_result']==1]

no_hearing_problem=df[df['test_result']==0]

plt.scatter(hearing_problem['age'],hearing_problem['physical_score'],color='red',marker=".")
plt.scatter(no_hearing_problem['age'],no_hearing_problem['physical_score'],color="green",marker=".")
#plt.show()

x=df.drop('test_result',axis=1)
y=df['test_result']

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23123123)

model=LogisticRegression()
model.fit(x_train,y_train)

output=model.predict(pd.DataFrame([[60,36]],columns=['age','physical_score']))
print(output)

y_pred=model.predict(x_test)
confusion=confusion_matrix(y_test,y_pred)
print("confusion matrix")
print(confusion)


accuracy=accuracy_score(y_test,y_pred)
print("Accuracy score:")
print(accuracy)

precision=precision_score(y_test,y_pred)
print("Precision :")
print(precision)

f1_score=f1_score(y_test,y_pred,average='weighted')
print("F1 score:")
print(f1_score)

