#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[8]:


boston=load_boston()

x=pd.DataFrame(boston.data,columns=boston.feature_names)

y=pd.DataFrame(boston.target,columns=['target'])

# for normalising the data we use minmax scaler method 

x=(x-np.min(x))/(np.max(x)-np.min(x))


# In[18]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)


# In[23]:


# create the liner regression model

linerregx=LinearRegression()
linerregx.fit(xtrain,ytrain)

accurecy=linerregx.score(xtest,ytest)

print('accurecy of the model is - ', accurecy*100)


# In[ ]:




