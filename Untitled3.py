#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df_mlb = pd.read_csv('mlb_elo_latest.csv')


# In[3]:


df_mlb


# In[1]:


# There are a lot of different columns that can help predict the winner of a game in baseball. 
# The columns Elo1_pre, Elo2_pre, Elo1_post, Elo2_post or Score1 and Score2 for example. 
# Like the pitcher1 column and the pitcher2 column can be used to predict the best chance of a team winning. 
# The pitchers are responsible for a lot of factors in the game of baseball and thus have more influence over the game. 
# So these 2 columns are best equipped to handle predictions. 
# The pitchers give the best chance to win the game; if a pitcher is on point, then the other team can't win. 
# So, if your roster =is filled with pitchers that are consistent and efficient, you have the best chance of winning. 
# In real life, coaches and teams could use predictive models to study opponents and make better strategies for upcoming matches.


# In[2]:


df_mlb.columns = df_mlb.columns.str.strip()


# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits import mplot3d


# In[4]:


X = df_mlb[['pitcher1_rgs', 'pitcher2_rgs']].values.reshape(-1, 2)


# In[ ]:




