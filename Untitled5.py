#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import statsmodels.api as sm


# In[29]:


mlb = pd.read_csv('mlb_elo_latest.csv')


# In[30]:


print(mlb.head())
print(mlb.describe())


# In[31]:


mlb_filled = mlb.fillna(0)


# In[32]:


columns_to_fill = ['elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'score1', 'score2','elo1_post']
mlb_filled[columns_to_fill] = mlb_filled[columns_to_fill].fillna(-1)


# In[33]:


y = mlb_filled['score1']
X = mlb_filled[['elo1_pre', 'elo_prob1', 'elo1_post']]


# In[34]:


X_np = X.to_numpy()
y_np = y.to_numpy()


# In[35]:


X_np = sm.add_constant(X_np)


# In[36]:


model = sm.OLS(y_np, X_np).fit()


# In[37]:


print(model.summary())


# In[ ]:




