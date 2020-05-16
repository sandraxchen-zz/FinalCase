#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ## Subscribers

# In[2]:


subscribers = pd.read_pickle('data/subscribers')

subscribers.head()


# In[116]:


subscribers_columns = ['subid', 'retarget_TF', 'monthly_price', 'discount_price',
                      'creation_until_cancel_days', 'cancel_before_trial_end', 'revenue_net', 'join_fee', 'paid_TF', 'refund_after_trial_TF']


# In[8]:


#nas

subscribers.isna().sum()


# In[5]:


subscribers.shape


# In[36]:


subscribers['attribution_technical'].value_counts()
# this is probably last interaction


# In[37]:


subscribers['attribution_survey'].value_counts()
# tv comes up here because you can't track tv


# ## Engagement

# In[6]:


engagement = pd.read_pickle('data/engagement')

engagement.head()


# In[72]:


engagement.shape


# In[77]:


# number of users:
len(engagement.subid.drop_duplicates())


# ## Cust Service Reps

# In[4]:


customer_service_reps = pd.read_pickle('data/customer_service_reps')

customer_service_reps.head()


# In[116]:


customer_service_reps.shape


# In[123]:


len(customer_service_reps.subid.drop_duplicates())


# In[668]:


customer_service_reps['customer_type'] = np.where(customer_service_reps['subid'].isin(subscribers['subid']), 'OTT + subscriber',
                                                 np.where(customer_service_reps['billing_channel'] == 'OTT', 'OTT only',
                                                         customer_service_reps['billing_channel']))

customer_service_reps[['subid', 'customer_type']].drop_duplicates().customer_type.value_counts()


# In[43]:


sub_billing = customer_service_reps[['subid', 'billing_channel']].drop_duplicates()

sub_billing.billing_channel.value_counts()


# In[693]:


subs_eachtable = customer_service_reps[['subid', 'billing_channel', 'current_sub_TF']].drop_duplicates().merge(subscribers[['subid', 'paid_TF']], on ='subid', how ='outer')
subs_eachtable = subs_eachtable.merge(engagement.groupby('subid').payment_period.agg('max').reset_index(), on ='subid', how ='outer')

subs_eachtable['engagement'] = np.where(subs_eachtable['payment_period'].isna(), 0, 1)
subs_eachtable['subscribers'] = np.where(subs_eachtable['paid_TF'].isna(), 0, 1)
subs_eachtable['cust_service'] = np.where(subs_eachtable['billing_channel'].isna(), 0, 1)

subs_eachtable = subs_eachtable.fillna('Null')

subs_eachtable['sub_type'] = np.where(subs_eachtable['billing_channel'].isin(['google', 'itunes']), 'App only',
                                     np.where(subs_eachtable['billing_channel'] == "Null", 'Subscriber,\nno cust service',
                                             np.where(subs_eachtable['subscribers'] == 0, 'OTT only',
                                                     'OTT and\nsubscriber')))

subs_eachtable.head()


# In[817]:


subs_typecount = subs_eachtable.pivot_table(index ='sub_type', columns = 'current_sub_TF', values ='subid', aggfunc ='count', fill_value = 0)

subs_typecount.columns = ['Churned', 'Active', 'Unknown']
#subs_typecount['Total'] = subs_typecount.sum(axis = 1)

subs_typecount


# In[818]:


print("Total customers: %d" % subs_typecount.sum().sum())


# In[725]:


plt.bar(subs_typecount.index, subs_typecount['Churned'], label = "Churned", color = '#7e57c2')
plt.bar(subs_typecount.index, subs_typecount['Active'], bottom = subs_typecount['Churned'], 
        label = "Active", color = '#c77025')
plt.bar(subs_typecount.index, subs_typecount['Unknown'], label = "Unknown", color = '#009688')

for i in range(0, len(subs_typecount)):
    plt.text(x= subs_typecount.index[i], y = subs_typecount['Total'][i], s = subs_typecount['Total'][i],
            verticalalignment='bottom', horizontalalignment='center')

plt.legend()
plt.ylabel('Customers')
plt.show()


# In[669]:


subs_typecount


# In[131]:


ids_analyze = subs_eachtable[subs_eachtable['sub_type'] == 'OTT and subscriber'].subid.values

print("Total ids to analyze: %d" % len(ids_analyze))


# ## Acquisition channel
# 
# * Attribution technical aligns more closely with the money that invested (no TV no radio)

# In[561]:


subscribers['attribution_technical'].value_counts().head(10)


# In[562]:


subscribers['attribution_survey'].value_counts().head(10)


# In[567]:


spend = pd.read_csv('data/advertisingspend.csv')
spend['date'] = pd.to_datetime(spend['date'], infer_datetime_format=True).dt.to_period('M')
spend.head()


# In[598]:


sum_spend = spend.drop('date', axis = 1).sum().sort_values(ascending=False)
sum_spend


# In[732]:


# spend total
plt.figure(figsize = (10,4))
plt.bar(sum_spend.index.str.replace('_', '\n').str.replace(' ', '\n'), sum_spend, color = '#7e57c2')
plt.ylabel('Total Spend')
plt.show()


# In[600]:


acq_channel = df[(df['refund_after_trial_TF'] == False) & (df['payment_period'] > 0)].groupby(['subid', 'account_creation_date']).payment_period.agg('max').reset_index()

acq_channel = acq_channel.merge(subscribers[['subid', 'attribution_technical', 'attribution_survey']], on = 'subid')

# aligning attribution channel to spend channels
acq_channel['attribution_technical2'] = np.where(acq_channel['attribution_technical'].isin(spend.columns), acq_channel['attribution_technical'], 
                                                 np.where(acq_channel['attribution_technical'].str.contains("organic"), "organic", 'other'))

acq_channel.head()


# In[583]:


len(acq_channel)


# In[602]:


# which channels didn't they spend in
acq_channel[~acq_channel['attribution_technical'].isin(spend.columns)].attribution_technical.value_counts()


# In[738]:


sum_acqchannel = acq_channel.attribution_technical2.value_counts()
sum_acqchannel2 = np.where(sum_acqchannel.index.isin(spend.columns), 0, sum_acqchannel)

plt.figure(figsize = (10,4))
plt.bar(sum_acqchannel.index.str.replace('_', '\n').str.replace(' ', '\n'),sum_acqchannel, 
        color = '#7e57c2', label = 'Spend Accounted')
plt.bar(sum_acqchannel.index.str.replace('_', '\n').str.replace(' ', '\n'),sum_acqchannel2, 
        color = '#c77025', label = 'No Spend')
plt.legend()
plt.ylabel('Customers')
plt.show()


# In[589]:


acq_channel.attribution_survey.value_counts().head(10)
# survey ads tv and radio


# In[763]:


acq_channel['month'] = acq_channel['account_creation_date'].dt.to_period('M')

acq_month = acq_channel.groupby('month').subid.agg('count')


# In[773]:


acq_channel.pivot_table(index = 'month', columns ='attribution_technical2', values = 'subid', aggfunc ='count')


# In[776]:


# monthly acquisition
plt.figure(figsize = (10,4))
plt.plot(acq_month.index.astype(str), acq_month, color = '#7e57c2')
plt.ylabel('Customers')
plt.show()


# In[778]:


# monthly spend
plt.figure(figsize = (10,4))
plt.plot(acq_month.index.astype(str), spend.iloc[:,1:].sum(axis = 1),
        color = '#c77025')
plt.ylabel('Advertising Spend')
plt.show()


# ## Summary for Final Presentation

# In[831]:


def print_summary(df, date_col):
    print('Total Rows: %d' % len(df))
    print('Unique Customers: %d' % len(df.subid.drop_duplicates()))
    print('Min date: %s' % str(min(df[date_col])))
    print('Max date: %s' % str(max(df[date_col])))


# In[832]:


print('Customer Service Reps')
print_summary(customer_service_reps, 'account_creation_date')


# In[834]:


print('Subscribers')
print_summary(subscribers, 'account_creation_date')


# In[836]:


print('Engagement')
print_summary(engagement, 'date')

