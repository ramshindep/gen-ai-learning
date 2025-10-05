import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

#Load and prepare the dataset
df=pd.read_csv('Mall_Customers.csv')
df=df.drop(["CustomerId","Gender","Age"],axis=1)
df.rename(columns={"Annual Income (k$)":"Income","Spending Score (1-100)":"Spending"},inplace=True)

wss=[]
cluster_range=range(1,11)

for k in cluster_range:
    model=KMeans(n_clusters=k,init='k-means++',random_state=123)
    model.fit(df)
    wss.append(model.inertia_)
    
plt.figure(figsize=(8,5))
plt.plot(cluster_range,wss,marker='o')
plt.title("Elbow Method For Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("wss(Inertia)")
plt.grid(True)
plt.show()