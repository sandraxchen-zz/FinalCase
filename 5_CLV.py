#!/usr/bin/env python
# coding: utf-8

# * Revenue = previous revenue + future revenue based on churn probability
# * CAC = average CAC per channel
# * CLV = Revenue - CAC
# 
# 
# Discount rate: 10%
# 
# Assumptions:
# * Customers will pay monthly price in future (not discount price)
# * Revenue from first period was revenue_net
# * Churn probability is the same in every period

# In[112]:


# read in libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[113]:


# discount rate at 10% annually
# discount renews every 4 months
r = 0.1 / 3


# In[117]:


# read in data
subscribers = pd.read_pickle('data/subscribers')

channelcac = pd.read_pickle('channel_cac')
churnpred = pd.read_pickle('current_churnpred')

spend = pd.read_csv('data/advertisingspend.csv')
spend['month'] = pd.to_datetime(spend['date'], infer_datetime_format=True).dt.to_period('M')
spend = spend.drop('date', axis = 1)


# In[120]:


channelcac


# In[118]:


# join the tables together
df_clv = churnpred.merge(subscribers[['subid', 'attribution_technical', 'revenue_net', 'monthly_price', 'discount_price']], 
                         on = 'subid')

df_clv = df_clv.merge(channelcac.reset_index()[['attribution_technical','cac']], 
                      on ='attribution_technical', how ='left')


# In[119]:


df_clv.isna().sum()


# In[18]:


df_clv.describe()


# In[121]:


# cleaning up the table

# all channels that aren't account for, make cac 0
df_clv['cac'] = df_clv['cac'].fillna(0)

# if revenue net is null, then fill in with the mean value
df_clv['revenue_net'] = df_clv['revenue_net'].fillna(df_clv['revenue_net'].mean())

# channel names: group organic and other together
df_clv['attribution_technical'] = np.where(df_clv['attribution_technical'].isin(spend.columns), df_clv['attribution_technical'], 
                                                 np.where(df_clv['attribution_technical'].str.contains("organic"), "organic", 'other'))

df_clv.head()


# In[122]:


# calculate clv
df_clv['future_rev'] = (df_clv['monthly_price'])* ((1+r)/(1+r-(1-df_clv['churn_prob']))) - (df_clv['monthly_price'])
df_clv['clv'] = df_clv['revenue_net'] + df_clv['future_rev'] - df_clv['cac']

clv_median = df_clv['clv'].median()
clv_mean = df_clv['clv'].mean()

df_clv.head()


# In[123]:


print(clv_median)
print(clv_mean)


# In[124]:


import matplotlib.ticker as ticker

formatter_x = ticker.FormatStrFormatter('$%1.0f')

fig, ax = plt.subplots(figsize=(6, 4))

ax.hist(df_clv['clv'], bins = 20, color = '#7e57c2')
ax.axvline(clv_median, color='#c77025', linestyle = "dashed")
ax.axvline(clv_mean, color='#009688', linestyle = "dashed")

ax.annotate('Median: ${:,.2f}'.format(clv_median), xy = (clv_median, 11900),
                  xytext=(5, -5),  # points vertical offset
                  textcoords="offset points", ha='left', va='bottom')
ax.annotate('Mean: ${:,.2f}'.format(clv_mean), xy = (clv_mean, 11900),
                  xytext=(5, -20), textcoords="offset points", ha='left', va='bottom')

ax.xaxis.set_major_formatter(formatter_x)
ax.set_ylabel('Count') 
ax.set_xlabel('CLV') 
plt.show()


# In[125]:


df_clv['clv'].describe()

