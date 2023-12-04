#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[28]:


mlb = pd.read_csv('mlb_elo_latest.csv')


# In[31]:


print(mlb.head())
print(mlb.describe())


# In[47]:


mlb_filled = mlb.fillna(0)


# In[48]:


columns_to_fill = ['elo1_pre', 'elo2_pre', 'elo_prob1', 'elo_prob2', 'score1', 'score2']
mlb[columns_to_fill] = mlb[columns_to_fill].fillna(-1)


# In[49]:


print(mlb.head())
print(mlb.describe())


# In[50]:


features = ['elo_prob1', 'elo_prob2']
target = ['score1', 'score2']

X = mlb[features] 
y = mlb[target] 


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[52]:


model = LinearRegression()  
model.fit(X_train, y_train)


# In[53]:


predictions = model.predict(X_test)


# In[54]:


mse = mean_squared_error(y_test, predictions)  
r2 = r2_score(y_test, predictions)


# In[55]:


print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

