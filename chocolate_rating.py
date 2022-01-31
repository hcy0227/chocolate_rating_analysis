#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# # install scikit learn

# In[4]:


get_ipython().system('pip install sklearn')


# In[5]:


# load data
df = pd.read_csv('chocolate_ratings.csv')


# In[6]:


df.head(10)


# In[7]:


df.info()


# In[8]:


df.shape


# In[9]:


type(df['Cocoa Percent'][0])


# In[10]:


type(df['Rating'][0])


# In[11]:


df.isna().sum()


# In[12]:


df['Company (Manufacturer)'].value_counts()


# In[13]:


df['Company Location'].value_counts()


# In[14]:


sns.set(rc={"figure.figsize":(20,10)})
ax = sns.countplot(x='Company Location', data=df, order=df['Company Location'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')


# In[15]:


df['Country of Bean Origin'].value_counts()


# In[16]:


ax = sns.countplot(x='Country of Bean Origin', data=df, order=df['Country of Bean Origin'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


# In[17]:


df['Rating'].value_counts()


# In[18]:


ax = sns.countplot(x = 'Rating', data = df)


# In[19]:


df['Cocoa Percent'] = df['Cocoa Percent'].str.replace('%', '') # remove '%' from the Cocoa Percent columns
df['Cocoa Percent'] = df['Cocoa Percent'].astype(float) # parse to float


# In[20]:


df['Cocoa Percent'].dtypes


# In[21]:


df.head()


# In[22]:


df['Cocoa Percent'].value_counts()


# In[23]:


dfax = sns.countplot(x = 'Cocoa Percent', data = df)


# In[24]:


df['Cocoa Percent'].hist() # histogram graph for Cocoa Percent


# In[25]:


df.plot.hexbin(x='Rating', y='Cocoa Percent', gridsize=15, sharex=False)


# In[26]:


sns.pairplot(data=df[['Cocoa Percent', 'Rating']])


# In[27]:


df.dtypes


# In[28]:


# ingredients of top rated chocolates
df['Ingredients'].value_counts()

df['Ingredients'].value_counts().plot(kind='bar')
plt.title('Ingredients - Top rated chocolates')
plt.grid()
plt.show()


# In[63]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

for col in df.columns:
    df[col] = enc.fit(df[col]).transform(df[col])


# In[64]:


df.dtypes


# In[65]:


df.corr()


# In[66]:


# To help us calculate cramers_v from above reference.
# A 1 means highly strong associated, a 0 means not
def cramers_v(x, y):
    import scipy.stats as ss
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


# In[67]:


cramers_v(df['Country of Bean Origin'], df['Rating'])


# In[68]:


cramers_v(df['Cocoa Percent'], df['Rating'])


# In[69]:


# Make an empty dataframe
df_cramers = pd.DataFrame(index = df.columns, columns = df.columns)


# In[70]:


for col in df_cramers.columns:
    for row in df_cramers.index:
        df_cramers[col][row] = cramers_v(df[col], df[row])


# In[71]:


sns.heatmap(df_cramers.astype(float))

