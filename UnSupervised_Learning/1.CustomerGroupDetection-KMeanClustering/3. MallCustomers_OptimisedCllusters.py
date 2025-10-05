import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans    

df=pd.read_csv('Mall_Customers.csv')
df=df.drop(["CustomerId","Gender","Age"],axis=1)
df=df.rename(columns={"Annual Income (k$)":"Income","Spending Score (1-100)":"Spending"},inplace=True)


KMeans_final=KMeans(n_clusters=5,random_state=123)
df["Cluster"]=KMeans_final.fit_predict(df)

plt.figure(figsize=(8,5))
sns.scatterplot(x='Income',y='Spending',hue='Cluster',data=df,palette='Set2',s=100)
plt.title('Customer Segemnts by Income and Spending ')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()

