import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')

data=pd.read_csv('Mall_Customers.csv',index_col='CustomerID')
data.head()
data.info()
data.describe()
data.isnull()
data.drop_duplicates(inplace=True)
x=data.iloc[:,[2,3]].values
print(x)
data.isnull().sum()
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


plt.figure(figsize=(10,5))
sns.lineplot(range(1,11), wcss, color='red')
plt.title('elbow method')
plt.xlabel('no. of clusters')
plt.ylabel('wcss')
plt.show()

kmeans=KMeans(n_clusters=5, init='k-means++', random_state=42)
y_pred=kmeans.fit_predict(x)
plt.figure(figsize=(10,6))
for i in range(5):
    plt.scatter(x[y_pred==i,0],x[y_pred==i,1],label='cluster'+str(i+1))
    plt.legend()
plt.title('Cluster of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()