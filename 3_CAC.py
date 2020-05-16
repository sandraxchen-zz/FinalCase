#!/usr/bin/env python
# coding: utf-8

# * Calculate Marginal CAC
# 
# Methodology:
# * As attribution is only available for subscriber dataset, attribution per channel was extrapolated to the whole customer base.
# 
# Definition of acquisition:
# * Made it past trial
# * Didn't request refund
# * Had revenue > 0
# 
# Assumptions:
# * Acquisition channel of subscriber base is representative of all customers
# * 20% of people who make it past trial request a refund for subscriber base. Assume this holds true for all customers.

# In[1]:


# read in libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[2]:


# read in data
subscribers = pd.read_pickle('data/subscribers')
customer_service_reps = pd.read_pickle('data/customer_service_reps')

spend = pd.read_csv('data/advertisingspend.csv')
spend['month'] = pd.to_datetime(spend['date'], infer_datetime_format=True).dt.to_period('M')
spend = spend.drop('date', axis = 1)


# In[3]:


customer_service_reps['month'] = customer_service_reps['account_creation_date'].dt.to_period('M')

subscribers['month'] = subscribers['account_creation_date'].dt.to_period('M')

# conversion defined as customer who made past trial, didn't cancel after trial, and had revenue > 0
subscribers['conversion'] = np.where((subscribers['cancel_before_trial_end'] == False) | (subscribers['revenue_net'] == 0) | (subscribers['refund_after_trial_TF'] == True),
                                    False, True)


# In[3]:


customer_service_reps.head()


# In[53]:


subscribers.groupby(['cancel_before_trial_end','conversion']).subid.count()


# In[54]:


refund_pct = len(subscribers[(subscribers['conversion'] == False) & (subscribers['cancel_before_trial_end'] == True)]) / len(subscribers[subscribers['cancel_before_trial_end'] == True])

refund_pct


# In[4]:


# acquisition from subscribers table
sub_acq = subscribers.loc[:,['subid','month','cancel_before_trial_end','conversion', 'attribution_technical']]

sub_acq['past_trial_sub'] = sub_acq['conversion']
sub_acq = sub_acq.drop('cancel_before_trial_end', axis = 1)

# change acquisition channel to match the spend data
sub_acq['attribution_technical'] = np.where(sub_acq['attribution_technical'].isin(spend.columns), sub_acq['attribution_technical'], 
                                                 np.where(sub_acq['attribution_technical'].str.contains("organic"), "organic", 'other'))

sub_acq.head()


# In[5]:


# acquisition for csr table
total_acq = customer_service_reps.groupby(['subid','month', 'trial_completed_TF', 'revenue_net_1month']).payment_period.max().reset_index()
total_acq['past_trial_csr'] = np.where((total_acq['trial_completed_TF'] == False) | (total_acq['payment_period'] == 0) | (total_acq['revenue_net_1month'] == 0), 
                                       False, True)
total_acq = total_acq.drop(['trial_completed_TF', 'payment_period', 'revenue_net_1month'], axis = 1)

total_acq.head(10)


# In[61]:


print(len(sub_acq))
print(len(total_acq))


# In[6]:


# merge csr and sub tables
acq_join = sub_acq.merge(total_acq, how ='outer', on =['subid', 'month'])

acq_join['past_trial'] = np.where(acq_join['past_trial_sub'].isna(), acq_join['past_trial_csr'], acq_join['past_trial_sub'])

acq_join['sub_flg'] = np.where(acq_join['past_trial_sub'].isna(), False, True)
acq_join['csr_flg'] = np.where(acq_join['past_trial_csr'].isna(), False, True)

acq_join = acq_join.drop(['past_trial_csr', 'past_trial_sub'], axis = 1)

acq_join.head()


# In[63]:


# monthly acquisition (subscribers only)
month_sub_acq = sub_acq[sub_acq['conversion'] == True].groupby('month').subid.agg('count')

plt.figure(figsize = (10,4))
plt.plot(month_sub_acq.index.astype('str'), month_sub_acq)
plt.show()


# In[64]:


spend_total = spend.set_index('month').sum(axis = 1)

plt.figure(figsize = (10,4))
plt.plot(spend_total.index.astype('str'), spend_total)
plt.show()


# ### Extrapolation

# In[7]:


# another method - look only at sign ups

extrapolate_factor = acq_join[acq_join['past_trial'] == True].groupby('month').sub_flg.agg(['sum', 'count']).reset_index()
extrapolate_factor['extrapolate'] = extrapolate_factor['count'] / extrapolate_factor['sum']
extrapolate_factor = extrapolate_factor[['month','extrapolate']]

extrapolate_factor


# In[8]:


# For each month and each channel, calculate how many customers signed up for trial
channel_sub = acq_join[(acq_join['sub_flg'] == True) & (acq_join['past_trial'] == True)].groupby(['month','attribution_technical']).subid.agg(['count']).reset_index()

channel_sub = channel_sub.merge(extrapolate_factor, on='month')

channel_sub['total_acq'] = channel_sub['count'] * channel_sub['extrapolate']

# Ad spend
channel_sub = channel_sub.merge(spend.melt('month', value_name = 'spend', var_name ='attribution_technical'), 
                                on = ['month', 'attribution_technical'])
channel_sub['spend'] = channel_sub['spend'].astype(int)

# Calculate CAC
channel_sub['cac'] = channel_sub['spend'] / channel_sub['total_acq']
channel_sub.round(2).head()

channel_sub.round(2).head()


# In[104]:


# Average CAC:
average_cac = channel_sub.groupby('attribution_technical')[['spend','total_acq']].agg('sum')
average_cac['cac'] = average_cac['spend'] / average_cac['total_acq']

average_cac.round(2)


# In[105]:


# save to pickle for clv calculation
average_cac.to_pickle('channel_cac')


# In[106]:


# Total Average cac

totalavgcac = sum(average_cac['spend']) / sum(average_cac['total_acq'])

totalavgcac


# In[107]:


# create bar plot to visualize average cac

N = len(average_cac)

fig, ax = plt.subplots(figsize=(7, 5))
plt.box(False)

ind = np.arange(N)    # the x locations for the groups
width = 0.80        # the width of the bars

p1 = ax.bar(average_cac.index, average_cac['cac'], 
            width, bottom=0, color = '#7e57c2')

ax.axhline(totalavgcac, color='#c77025', linestyle = "dashed")

#ax.set_title(title)
ax.set(xlabel='', ylabel= 'Average CAC')
ax.set_xticks(ind)
ax.set_xticklabels(average_cac.index.str.replace('_', '\n').str.replace(' ', '\n'))

vals = ax.get_yticks()
ax.set_yticklabels(['${:,.0f}'.format(x) for x in vals])
ax.autoscale_view()

# cluster label
def autolabel(rects):
  for rect in rects:
      height = rect.get_height()
      ax.annotate('${:,.2f}'.format(height),
                  xy=(rect.get_x() + rect.get_width() / 2, height),
                  xytext=(0, 2),  # points vertical offset
                  textcoords="offset points",
                  ha='center', va='bottom')

#average label
ax.annotate('Total Avg: \n${:,.2f}'.format(totalavgcac), xy = (ind[-1], totalavgcac),
                  xytext=(15, 2),  # points vertical offset
                  textcoords="offset points",
                  ha='left', va='bottom', fontweight = 'bold')

autolabel(p1)

plt.show()


# In[112]:


# Total Average cac

totalavgcac = sum(average_cac['spend']) / sum(average_cac['total_acq'])

totalavgcac


# In[113]:


print('Acquisitons from subscriber: %.2f' % len(sub_acq[sub_acq['conversion'] == True]))
print('Total (estimated) acquisitions: %.2f' % average_cac['total_acq'].sum())
print('%% (estimated): %.2f' % (len(sub_acq[sub_acq['conversion'] == True]) / average_cac['total_acq'].sum()))

print('Trial signups subs/total: %.2f' % (len(sub_acq) / len(acq_join)))

# My total estimated acquisitions is too low


# In[114]:


# Alternative method: divide spend
# used as a comparison to make sure my calculations are right. Need to not do divided way so I can calculate marginal.

# Get spend
dividespend = spend.melt('month', value_name = 'spend', var_name ='attribution_technical')

# divide by extrapolation factor
dividespend = dividespend.merge(extrapolate_factor, on='month')
dividespend['adj_spend'] = dividespend['spend'] / dividespend['extrapolate']

# get number of converted customers
dividespend = dividespend.merge(acq_join[acq_join['conversion'] == True].groupby(['month','attribution_technical']).subid.agg(['count']).reset_index())
dividespend['cac'] = dividespend['adj_spend'] / dividespend['count']

# Average CAC:
divideaverage_cac = dividespend.groupby('attribution_technical')[['adj_spend','count']].agg('sum')
divideaverage_cac['cac'] = divideaverage_cac['adj_spend'] / divideaverage_cac['count']

divideaverage_cac


# In[111]:


divideaverage_cac['adj_spend'].sum() / divideaverage_cac['count'].sum()


# ## Marginal CAC
# 
# * Optimal spend defined as point of diminishing returns?

# In[115]:


# total average cac

totalavg_cac = average_cac.spend.sum() / average_cac.total_acq.sum()
print('Total average CAC: %.2f' % totalavg_cac)


# In[116]:


marginal_cac = channel_sub[['month','attribution_technical','spend','total_acq']]

#remove June because of bias
marginal_cac = marginal_cac[marginal_cac['month'] != '2019-06']

# if 2 months have same spend, then take average
#marginal_cac = marginal_cac.groupby(['attribution_technical','spend']).total_acq.mean().reset_index()

marginal_cac = marginal_cac.sort_values(['attribution_technical', 'spend'])

marginal_cac['month_cac'] = marginal_cac['spend'] / marginal_cac['total_acq']
marginal_cac['marginal_acq'] = marginal_cac['total_acq'] - marginal_cac.groupby('attribution_technical')['total_acq'].shift(1, fill_value = 0)
marginal_cac['marginal_spend'] = marginal_cac['spend'] - marginal_cac.groupby('attribution_technical')['spend'].shift(1, fill_value = 0)
marginal_cac['marginal_cac'] = marginal_cac['marginal_spend'] / marginal_cac['marginal_acq']

marginal_cac[marginal_cac['attribution_technical'] == 'facebook'].round(2)


# In[117]:


# write to CSV for easier pivot table analysis
marginal_cac.to_csv('marginal_cac.csv')


# In[118]:


# plot monthly cac per channel

import matplotlib.ticker as ticker

formatter_y = ticker.FormatStrFormatter('$%1.2f')
formatter_x = ticker.FormatStrFormatter('$%1.0f')
    
fig = plt.figure(figsize=(16,7))

c = 0

for channel in marginal_cac['attribution_technical'].drop_duplicates():
    
    df = marginal_cac[marginal_cac['attribution_technical'] == channel]
    ax = fig.add_subplot(2,4, c+1)
    
    ax.axhline(average_cac.loc[average_cac.index == channel, 'cac'][0], 
               linestyle = "dashed", color = '#c77025', label = 'Channel avg CAC')
    ax.axhline(totalavgcac, linestyle = "dashed", color = '#009688', label = 'Overall avg CAC')
    p1 = ax.scatter(df['spend'], df['spend'] / df['total_acq'], color = '#7e57c2', label = 'Monthly CAC')
    ax.set_title(channel,size = 14)
    
    ax.yaxis.set_major_formatter(formatter_y)
    ax.xaxis.set_major_formatter(formatter_x)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    # put ylabel on graphs on left
    if (c % 4) == 0:
        ax.set_ylabel('CAC', size = 14)
    
    # put xlabel on last 3 graphs
    if c > 8-5:
        ax.set_xlabel('Spend', size = 14)    
    
    c += 1

#ax = fig.add_subplot(3,3, c+1)    
ax.legend(loc=9, bbox_to_anchor=(0.5,-0.2))   

plt.tight_layout()
plt.show()


# * Affiliate - High investment, 17600 is good
# * Brand SEM intent google - low spend around 21300 is good, dimminishing returns
# * Email - always performs lower than average. Recommend pulling investment from it.
# * Email blast - performs well at low spend, less than 29800
# * Facebook - performing well at high spend, even at the current highest spend of 60000. Can try spending more.
# * Pinterest - performs well at low amount of spend, around 6300
# * Referral - medium amount of investment
# 
# Low investment in Facebook at the moment. Too much investment in email.
# Current strategy has too much money in email, but more should be invested into Facebook and affiliate.
# 
