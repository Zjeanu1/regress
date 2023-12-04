#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[13]:


df = pd.read_csv('Emissions_Canada.csv')


# In[14]:


print(df.head())
print(df.describe())


# In[17]:


features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)']
target = ['CO2 Emissions(g/km)']

X = df[features]
y = df[target]


# In[18]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[19]:


model = LinearRegression()  
model.fit(X_train, y_train)


# In[20]:


predictions = model.predict(X_test)


# In[21]:


mse = mean_squared_error(y_test, predictions)  
r2 = r2_score(y_test, predictions)


# In[22]:


print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[ ]:




