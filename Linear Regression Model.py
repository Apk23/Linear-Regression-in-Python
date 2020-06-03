#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[2]:


data = pd.read_csv('clean.csv')


# In[3]:


data.describe()


# In[4]:


x = DataFrame(data, columns = ['production_budget_usd'])
y = DataFrame(data, columns = ['worldwide_gross_usd'])


# In[5]:


plt.scatter(x, y, alpha= 0.3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)
plt.show()


# In[6]:


regression = LinearRegression()
regression.fit(x,y)


# In[7]:


plt.scatter(x, y, alpha= 0.3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production Budget $')
plt.ylabel('Worldwide Gross $')
plt.ylim(0, 3000000000)
plt.xlim(0, 450000000)

plt.plot(x, regression.predict(x) , color = "green" , alpha = 0.9, linewidth = 4)
plt.show()


# In[ ]:





# In[ ]:




