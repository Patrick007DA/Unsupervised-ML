#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os


# In[2]:


import warnings

warnings.filterwarnings('ignore')


# In[3]:


os.chdir(r'C:\Users\Pratik G Ratnaparkhi\Desktop\IVY Python')
path_data = os.getcwd()
data = pd.read_csv('Mall_Customers.csv')


# In[4]:


data.head()


# In[5]:


#Renaming The Columns name
data.rename(columns={"Genre":"Gender"
                     ,"Annual Income (k$)":"Annual_Income",
                     "Spending Score (1-100)":"Spending_Score"},inplace=True)
data.head()


# In[6]:


#Avg Score spends by Gender
Avg_score = data[['Gender','Spending_Score']].groupby(['Gender'],as_index=False).mean()
Avg_score


# In[ ]:





# In[7]:


#Avg Age Of Male And Female
Avg_Age = data[['Gender','Age']].groupby(['Gender'],as_index=False).mean()
Avg_Age


# In[8]:


#we will do some basic Data Visualization


# In[9]:


#Income wrt Genre
g = sns.FacetGrid(data,col='Gender')
g.map(plt.hist,'Annual_Income',bins=20)


# In[10]:


#Ploting pairplot
sns.pairplot(data[['Age','Annual_Income','Spending_Score']])


# In[11]:


#Checking For Outliers using Box-Plot
plt.subplot(1,3,1)
plt.boxplot(data['Age'])
plt.title('Age')
plt.subplot(1,3,2)
plt.boxplot(data['Annual_Income'])
plt.title('Annual_Income')
plt.subplot(1,3,3)
plt.boxplot(data['Spending_Score'])
plt.title('Spending_Score')
plt.tight_layout()


# In[12]:


#droping Gender feature
x = data.drop(['Gender','CustomerID'],axis=1)
y = data['Gender']


# In[13]:


# As we are clustering on the basis of Age, Income and Score 
# We will drop gender column


# In[14]:


print(x.head())

print(y.head())


# In[15]:


#Exploratory Data Analysis
print(x.describe(include="all"))


# In[16]:


#checking missing values
print(x.isna().sum())


# In[17]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
x_scaled = scaler1.fit_transform(x)


# In[18]:


x_scaled = pd.DataFrame(x_scaled)


# In[19]:


kmeans=KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
       random_state=0, tol=0.0001, verbose=0)
x_scaled1 = x_scaled._get_numeric_data().dropna(axis=1)
kmeans.fit(x_scaled1)
predict=kmeans.predict(x_scaled1)


# In[20]:


kmeans.cluster_centers_


# In[21]:


kmeans.inertia_


# In[22]:


inertias = []
mapping = {}

for k in range(1,10):
    kmeansmodel = KMeans(n_clusters=k, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeansmodel.fit(x_scaled)
    inertias.append(kmeansmodel.inertia_)
    mapping[k] = kmeansmodel.inertia_


# In[23]:


#ploting Elbow plot to know the number of clusters 
plt.plot(range(1, 10), inertias)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertias')
plt.show()


# In[56]:


#Checking Silhouette score
from sklearn.metrics import silhouette_score
range_k = [2,3,4,5,6,7]
for i in range_k:
    kmeans = KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=i, n_init=10, n_jobs=1, precompute_distances='auto',
       random_state=0, tol=0.0001, verbose=0)
    kmeans.fit(x_scaled)
    clabel = kmeans.labels_
    score = silhouette_score(x_scaled,clabel)
    print("For n_clusters={0}, the silhouette score is {1}".format(i, score))


# In[73]:


#From plot lets make a final model with k=4
x_scaled = pd.DataFrame(x_scaled)
kmeans=KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=4, n_init=10, n_jobs=1, precompute_distances='auto',
       random_state=0, tol=0.0001, verbose=0)
x_scaled1 = x_scaled._get_numeric_data().dropna(axis=1)
kmeans.fit(x_scaled1)
predict=kmeans.predict(x_scaled1)


# In[74]:


x_scaled = pd.DataFrame(x_scaled)


# In[75]:


x_scaled1 = x_scaled['clusters']=pd.Series(predict,index=x_scaled.index)
x_scaled1


# In[76]:


x['cluster_id'] = kmeans.labels_
x.head()


# In[77]:


sns.boxplot(x='cluster_id', y='Spending_Score', data=x)


# In[ ]:




