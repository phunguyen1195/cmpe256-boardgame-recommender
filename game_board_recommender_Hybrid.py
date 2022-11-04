#!/usr/bin/env python
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import os, io
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR


# In[35]:


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_theme = pd.read_csv('subcategories.csv')
df_game = pd.read_csv('games.csv')
# reader = Reader(line_format="user item rating", sep=",", rating_scale=(1,10))


# In[36]:


df_game


# In[37]:


df_theme_game = pd.merge(df_game, df_theme, on="BGGId")


# In[38]:


df_theme_game = df_theme_game.rename(columns={"BGGId": "GameID"})


# In[39]:


df_theme_game = df_theme_game.drop(columns=['Name', 'Description'])


# In[41]:


df_train =  pd.merge(df_train, df_theme_game, on="GameID")
df_test = pd.merge(df_test, df_theme_game, on="GameID")


# In[42]:


X_train, X_test, y_train, y_test = train_test_split(
    df_train.drop(columns=['Rating']), df_train['Rating'], test_size=0.2, random_state=42)


# In[43]:


pipe = make_pipeline(MinMaxScaler())


# In[44]:


X_train = pipe.fit_transform(X_train)
X_train = pd.DataFrame(X_train)
X_test = pipe.transform(X_test)


# In[49]:


svr = SVR(C=1.0, epsilon=0.2)
#forest_reg.fit(X_train, y_train)


# In[ ]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(svr, X_train, y_train,
                         scoring='neg_mean_absolute_error', cv=2, n_jobs=-1)
forest_mae_scores = -score
print(forest_mae_scores)
f = open("best_params.txt", "w")
f.write(str(grid_search.best_score_))
f.write(str(grid_search.best_params_))
f.close()


# In[ ]:




