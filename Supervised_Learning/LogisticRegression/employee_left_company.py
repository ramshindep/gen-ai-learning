import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,f1_score

df=pd.read_csv('employee_data.csv')

print(df['Age'].corr(df['Employee_Left']))
print(df['Hours_Worked_Per_Week'].corr(df['Employee_Left']))
print(df['Salary'].corr(df['Employee_Left']))

employee_left=df[df['Employee_Left']==1]
employeee_stayed=df[df['Employee_Left']==0]

plt.scatter(employee_left['Age'],employee_left['Salary'],color='red',marker=".")
plt.scatter(employeee_stayed['Age'],employeee_stayed['Salary'],color="green",marker=".")
#plt.show()

x=df.drop('Employee_Left',axis=1)
y=df['Employee_Left']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=23123123)

model=LogisticRegression()
model.fit(x_train,y_train)

output=model.predict(pd.DataFrame([[35,40,80000]],columns=['Age','Hours_Worked_Per_Week','Salary']))
print(output)

y_pred=model.predict(x_test)
confusion_matrix=confusion_matrix(y_test,y_pred)
print("confusion matrix")
print(confusion_matrix)

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy score:")
print(accuracy)

precision=precision_score(y_test,y_pred)
print("Precision :")
print(precision)

f1_score=f1_score(y_test,y_pred,average='weighted')
print("F1 score:")
print(f1_score)