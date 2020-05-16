#!/usr/bin/env python
# coding: utf-8

# In[1]:


# read in libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats as st
import math


# In[2]:


# read in data
subscribers = pd.read_pickle('data/subscribers')
customer_service_reps = pd.read_pickle('data/customer_service_reps')

spend = pd.read_csv('data/advertisingspend.csv')
spend['month'] = pd.to_datetime(spend['date'], infer_datetime_format=True).dt.to_period('M')
spend = spend.drop('date', axis = 1)

customer_service_reps['month'] = customer_service_reps['account_creation_date'].dt.to_period('M')

subscribers['month'] = subscribers['account_creation_date'].dt.to_period('M')

# conversion defined as customer who made past trial, didn't cancel after trial, and had revenue > 1
subscribers['conversion'] = np.where((subscribers['cancel_before_trial_end'] == False) | (subscribers['revenue_net'] < 1) | (subscribers['refund_after_trial_TF'] == True),
                                    False, True)


# In[11]:


# confidence level 90%
# power 80%
alpha = 0.1
z_alpha = st.norm.ppf(1-(alpha/2))
print("z_alpha2: %.2f" % z_alpha)

beta = 1-0.8
z_beta = st.norm.ppf(1-beta)
print("z_beta: %.2f" % z_beta)

def n_optimal(p0 = p0, p1 = p1):
    p_hat = (p0 + p1) / 2
    delta = abs(p0 - p1)
    n_opt = ((z_alpha*math.sqrt(2*p_hat*(1-p_hat)) + z_beta * math.sqrt(p0*(1-p0) + p1 *(1-p1)))**2 ) / delta**2
    return n_opt


# ## Test plans
# * Only do November because high was only offered this month
# * Use one sample t-test, with base plan being population conversion rate

# In[33]:


# plans I'm interested in
plan_0 = 'base_uae_14_day_trial'
plan_1 = 'high_uae_14_day_trial'


# In[26]:


# filter subscribers to the plans and month I'm interested in
ab_basehigh = subscribers[(subscribers['plan_type'].isin([plan_0,plan_1])) & (subscribers['month'] =='2019-11')][['subid', 'plan_type', 'discount_price', 'conversion']]


# In[34]:


# summary count
ab_basehigh_count = ab_basehigh.groupby(['plan_type'])['conversion'].agg(['count','sum']).reset_index()
ab_basehigh_count['conversion'] = ab_basehigh_count['sum'] / ab_basehigh_count['count']

ab_basehigh_count


# In[31]:


ab_basehigh.groupby('plan_type').discount_price.mean()


# In[72]:


# conversion rate of 2 plans
p0 = ab_basehigh_count.loc[0,'conversion']
p1 = ab_basehigh_count.loc[1,'conversion']

print("p0: %.2f" % p0)
print("p1: %.2f" % p1)


# In[74]:


# calculating optimal sample size
n_opt = n_optimal(p0, p1)

print('Optimal sample size: %.2f' % n_opt)


# In[38]:


n = ab_basehigh_count.loc[1,'count']
z = (p1 - p0) / math.sqrt( (p0*(1-p0)) / n )
print("z-score: %.2f" % z)


# Reject Null hypothesis. Conversion rate is worse when offering the higher plan. Do not recommend increasing price.
# 
# However, the sample size is less than the optimal (325 vs 405).

# ## Test trial days
# * iTunes
# * no record of customers requesting refund after trial, so need to assume this is the same across both groups
# * only consider Nov and Dec because there was a high number of both 7 and 14 day trial. Before October, it was mostly 7 days. After, it was mostly 14 days.
# * Assume no difference between these 2 months (their conversion rates look the same)
# * 2 sample test because sample sizes are pretty similar

# In[7]:


ab_trial = customer_service_reps[customer_service_reps.billing_channel == 'itunes'].groupby(['subid','month', 'num_trial_days', 'revenue_net_1month']).payment_period.max().reset_index()

# convert if past 0 and has some revenue
ab_trial['conversion'] = np.where((ab_trial['payment_period'] > 0) & (ab_trial['revenue_net_1month'] > 0), True, False)

#filter to october and november
ab_trial = ab_trial[(ab_trial['month'] == '2019-12')]

#filter out 0 days
ab_trial = ab_trial[ab_trial['num_trial_days'] != 0]

ab_trial.head()


# In[12]:


ab_trial_count = ab_trial.groupby(['num_trial_days']).conversion.agg(['count','sum']).reset_index()
ab_trial_count['conversion'] = ab_trial_count['sum'] / ab_trial_count['count']
ab_trial_count


# In[13]:


p0 = ab_trial_count.loc[0,'conversion']
p1 = ab_trial_count.loc[1,'conversion']

n0 = ab_trial_count.loc[0,'count']
n1 = ab_trial_count.loc[1,'count']

print(round(p0,2))
print(round(p1,2))
print(n0)
print(n1)


# In[14]:


#calculate optimal sample size
n_optimal(p0, p1)


# In[15]:


# calculate 2 sample z stat
p_hat = (p0+p1)/2

z_trial = (p0 - p1) / math.sqrt(p_hat*(1-p_hat) * ((1/n0) + (1/n1)) )
print("z-score: %.2f" % z_trial)


# Reject Null hypothesis. Sample size is big enough.
