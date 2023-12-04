#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import statsmodels.api as sm


# In[32]:


df = pd.read_csv('Emissions_Canada.csv')


# In[33]:


print(df.head())
print(df.describe())


# In[34]:


df.columns = df.columns.str.strip()


# In[35]:


X = df[['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)']]
y = df['CO2 Emissions(g/km)']


# In[37]:


X_np = X.to_numpy()
y_np = y.to_numpy()


# In[38]:


X_np = sm.add_constant(X_np)


# In[39]:


model = sm.OLS(y_np, X_np).fit()


# In[40]:


print(model.summary())


# In[ ]:




