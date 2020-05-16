#!/usr/bin/env python
# coding: utf-8

# Only engagement data from period 0 and period 1 is available, so can only do churn modeling on that.
# Majority don't have a cancel date, so they only unengaged.

# In[101]:


# read in libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.metrics import auc, classification_report, f1_score, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier, export_graphviz


# In[2]:


# read in data
subscribers = pd.read_pickle('data/subscribers')
customer_service_reps = pd.read_pickle('data/customer_service_reps')
engagement = pd.read_pickle('data/engagement')

spend = pd.read_csv('data/advertisingspend.csv')
spend['month'] = pd.to_datetime(spend['date'], infer_datetime_format=True).dt.to_period('M')
spend = spend.drop('date', axis = 1)


# In[ ]:


# determine users to analyze
# we want people who passed their trial


# ### Prepare customers to analyze

# In[3]:


# find customers from subscribers past trial
# definition: finished trial, made revenue > 0, didn't request refund
# keep any na's for revenue_net
subids_subs = subscribers[(subscribers['cancel_before_trial_end'] == True) & ~(subscribers['revenue_net'] < 1) & (subscribers['refund_after_trial_TF'] == False)].subid.values

len(subids_subs)


# In[4]:


# i want to get a record of for each payment period - did they churn, renew or need to predict

# filter to OTT and past trial
subids_csr = customer_service_reps[(customer_service_reps['billing_channel'] == 'OTT')]

# filter to people who fit my definition of subscribing
# finished trial, made revenue > 0
subids_csr = subids_csr[(subids_csr['trial_completed_TF'] == True) & (subids_csr['revenue_net_1month'] >0)]

# only interested in periods past trial
subids_csr = subids_csr[subids_csr['payment_period'] > 0]
subids_csr = subids_csr[['subid','account_creation_date','current_sub_TF', 'payment_period', 'renew', 'last_payment', 'next_payment', 'cancel_date']]

# max period
subids_csr['max_period'] = subids_csr.groupby('subid').payment_period.transform('max')

# filter to period 1
subids_csr = subids_csr[subids_csr['payment_period'] == 1]

subids_csr['churn_flag'] = np.where((subids_csr['payment_period'] == subids_csr['max_period']) & (subids_csr['current_sub_TF']==True), 'current',
                                   np.where(subids_csr['payment_period'] != subids_csr['max_period'], '0', '1'))

subids_csr['last_payment'] = subids_csr['last_payment'].dt.date

subids_csr.head()


# In[5]:


subids = subids_csr[subids_csr['subid'].isin(subids_subs)][['subid', 'churn_flag', 'last_payment', 'cancel_date']]

subids.head()


# In[11]:


churnsubs_count = subids.churn_flag.value_counts()

churnsubs_count


# In[30]:


xlabel = ['Historic Customers\n(test/train set)', 'Current Customers\n(predict set)']

plt.bar(xlabel[0], churnsubs_count['0'], label = 'Renew', color = '#7e57c2')
plt.bar(xlabel[0], churnsubs_count['1'], bottom = churnsubs_count['0'], label = 'Churn', color = '#009688')
plt.bar(xlabel[1], churnsubs_count['1'], label = 'Predict', color = '#c77025')

plt.text(xlabel[0], churnsubs_count['0']/2, churnsubs_count['0'], ha='center', va='center', color = 'white', size = 14)
plt.text(xlabel[0], churnsubs_count['0'] + churnsubs_count['1']/2, churnsubs_count['1'], 
         ha='center', va='center', color = 'white', size = 14)
plt.text(xlabel[1], churnsubs_count['current']/2, churnsubs_count['current'], 
         ha='center', va='center', color = 'white', size = 14)

plt.legend()
plt.show()


# In[142]:


len(subids)


# ### Engagement data - descriptive analysis

# In[444]:


csr_churn = customer_service_reps.copy()

# adjust payment period for non-ott
csr_churn['payment_period'] = np.where(csr_churn['billing_channel'] == 'OTT', csr_churn['payment_period'],
                                      ((customer_service_reps['payment_period'] - 1) // 4) + 1)

csr_churn['max_period'] = csr_churn.groupby('subid').payment_period.transform('max')

# filter to customers who had revenue > 0 and revenue > 0
csr_churn = csr_churn[(csr_churn['max_period'] > 0) & (csr_churn['revenue_net_1month'] > 0)]

csr_churn = csr_churn[csr_churn['payment_period'] == 1]

#filter out current subs in period 1
csr_churn = csr_churn[~((csr_churn['current_sub_TF'] == True) & (csr_churn['max_period'] == 1))]

# make churn flag
csr_churn['churn_flag'] = np.where(csr_churn['max_period'] == 1, 1, 0)

csr_churn = csr_churn[['subid', 'billing_channel', 'churn_flag']]

csr_churn.head()


# In[445]:


csr_churn.groupby('billing_channel').churn_flag.mean()


# In[439]:


subids[subids['churn_flag'] != 'current'].churn_flag.astype(int).mean()


# In[433]:


csr_churn['churn_flag'].mean()


# In[430]:


len(csr_churn)


# In[395]:


engage_sub = engagement.merge(subids,on ='subid')

engage_sub['isnull'] = np.where(engage_sub['app_opens'].isna(), True, False)

engage_sub.groupby(['payment_period', 'isnull']).subid.count()


# Only Engagement period 3 has nulls so it should be excluded for fairness.

# ### Engagement Data - create model features

# In[396]:


# count of dates
df_engage = engagement[(engagement['payment_period'] > 0) & (engagement['payment_period'] < 3)].drop('payment_period', axis = 1)

df_engage = df_engage.groupby('subid').agg({'date': ['min', 'max'], 'app_opens': ['count', 'sum'],
                                           'cust_service_mssgs' : 'sum', 'num_videos_rated': 'sum',
                                           'num_videos_completed': 'sum', 'num_videos_more_than_30_seconds': 'sum',
                                           'num_series_started': 'sum'}).reset_index()
df_engage.columns = ['subid','date_min', 'date_max', 'active_days', 
                     'app_opens', 'cust_service_mssgs', 'num_videos_rated', 'num_videos_completed',
                    'num_videos_more_than_30_seconds', 'num_series_started']

df_engage['total_days'] = (df_engage['date_max'] - df_engage['date_min']).dt.days + 1

#creating the features i want
df_engage['active_pct'] = df_engage['active_days'] / df_engage['total_days']
df_engage['avg_app_opens'] = df_engage['app_opens'] / df_engage['total_days']
df_engage['avg_cust_service'] = df_engage['cust_service_mssgs'] / df_engage['total_days']
df_engage['avg_videos_completed'] = df_engage['num_videos_completed'] / df_engage['total_days']
df_engage['avg_videos_viewed'] = df_engage['num_videos_more_than_30_seconds'] / df_engage['total_days']
df_engage['avg_videos_rated'] = df_engage['num_videos_rated'] / df_engage['total_days']
df_engage['avg_series_started'] = df_engage['num_series_started'] / df_engage['total_days']
df_engage['pct_video_completed'] = df_engage['num_videos_completed'] / df_engage['num_videos_more_than_30_seconds']
df_engage['pct_video_rated'] = df_engage['num_videos_rated'] / df_engage['num_videos_more_than_30_seconds']
df_engage['pct_series_started'] = df_engage['num_series_started'] / df_engage['num_videos_more_than_30_seconds']
df_engage['vids_per_session'] = df_engage['num_videos_more_than_30_seconds'] / df_engage['app_opens']

# fill inf and nan
df_engage = df_engage.fillna(0)
df_engage['pct_video_rated'] = np.where(np.isinf(df_engage['pct_video_rated']), 1, df_engage['pct_video_rated'])
df_engage['pct_series_started'] = np.where(np.isinf(df_engage['pct_series_started']), 1, df_engage['pct_series_started'])
df_engage['vids_per_session'] = np.where(np.isinf(df_engage['vids_per_session']), df_engage['num_videos_more_than_30_seconds'], df_engage['vids_per_session'])

# drop what I don't want
df_engage = df_engage.drop(engagement.columns.values[2:-1], axis = 1)
df_engage = df_engage.drop(['date_min', 'date_max', 'active_days', 'total_days'], axis = 1)

df_engage.describe()


# In[397]:


# compare means to get a sense of whether it is correlated

df_engage.merge(subids, on ='subid').drop('subid', axis = 1).groupby('churn_flag').mean()


# ### Combine for model

# In[402]:


# combine model features
df_model = subids[['subid', 'churn_flag']].merge(df_engage,on ='subid')

#drop some highly correlated variables
df_model = df_model.drop(['avg_videos_completed', 'avg_videos_rated'], axis = 1) 

# customers to predict on
X_predict = df_model[df_model['churn_flag'] == 'current'].drop(['subid', 'churn_flag'], axis = 1)

# split to test train
X = df_model[df_model['churn_flag'] != 'current'].drop(['subid', 'churn_flag'],axis=1)
y = df_model[df_model['churn_flag'] != 'current']['churn_flag'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)

# scale
scaler = StandardScaler()

X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)
X_predict_scale = scaler.transform(X_predict)

X_predict.head()


# In[403]:


# check for correlation between predictor variables
X.corr()


# #### Linear Regression

# In[404]:


# linear regression

import statsmodels.api as sm
from scipy import stats

X2 = sm.add_constant(X_train_scale)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())


# In[405]:


# LASSO

# tuning hyperparameters
clf = Lasso()

# Tune the hyperparameter
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 100]}

# Grid Search
grid_search = GridSearchCV(clf, param_grid, cv=5, return_train_score=True, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_scale, y_train)

print('Grid Search Params')
print(grid_search.best_params_)

# don't do cross validation for linear
clf = grid_search.best_estimator_
clf.fit(X_train_scale, y_train)

y_pred_prob = clf.predict(X_test_scale)
#y_pred_prob = [1 if i > 1 else i for i in y_pred_prob] 

y_pred_train = clf.predict(X_train_scale).round()
#y_pred_train = [1 if i > 1 else i for i in y_pred_prob] 

y_pred = y_pred_prob.round()

print('Training accuracy: %.2f%%' % (accuracy_score(y_train, y_pred_train) * 100))
print('Test accuracy: %.2f%%' % (accuracy_score(y_test, y_pred) * 100))
print('AUC: %.2f%%' % (metrics.roc_auc_score(y_test,y_pred_prob) * 100))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

fpr_linear, tpr_linear, _ = roc_curve(y_test, y_pred_prob)


# In[406]:


# RIDGE

# tuning hyperparameters
clf = Ridge()

# Tune the hyperparameter
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 100]}

# Grid Search
grid_search = GridSearchCV(clf, param_grid, cv=5, return_train_score=True, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_scale, y_train)

print('Grid Search Params')
print(grid_search.best_params_)

# don't do cross validation for linear
clf = grid_search.best_estimator_
clf.fit(X_train_scale, y_train)

y_pred_prob = clf.predict(X_test_scale)
#y_pred_prob = [1 if i > 1 else i for i in y_pred_prob] 

y_pred_train = clf.predict(X_train_scale).round()
#y_pred_train = [1 if i > 1 else i for i in y_pred_prob] 

y_pred = y_pred_prob.round()

print('Training accuracy: %.2f%%' % (accuracy_score(y_train, y_pred_train) * 100))
print('Test accuracy: %.2f%%' % (accuracy_score(y_test, y_pred) * 100))
print('AUC: %.2f%%' % (metrics.roc_auc_score(y_test,y_pred_prob) * 100))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# #### Logistic Regression

# In[407]:


# tuning hyperparameters
clf = LogisticRegression(random_state=1)

# Tune the hyperparameter
param_grid = {'penalty': ['l1', 'l2'],
             'C': [0.001, 0.01, 0.1, 1, 100]}

# Grid Search
grid_search = GridSearchCV(clf, param_grid, cv=5, return_train_score=True, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_scale, y_train)

print('Grid Search Params')
print(grid_search.best_params_)

# apply to model to get scores
clf_gridsearch = grid_search.best_estimator_

scores = cross_validate(clf_gridsearch, X_train_scale, y_train, cv=5, scoring = ['accuracy', 'roc_auc'], return_train_score=True)

print('Cross Validation Results')
print('Train accuracy CV: %.4f' % scores['train_accuracy'].mean())
print('Test accuracy CV: %.4f' % scores['test_accuracy'].mean())
print('Test AUC: %.4f' % scores['test_roc_auc'].mean())


# In[408]:


clf = grid_search.best_estimator_
clf.fit(X_train_scale, y_train)
y_pred_prob = clf.predict_proba(X_test_scale)[:,1]
y_pred = clf.predict(X_test_scale)

print('Training accuracy: %.2f%%' % (clf.score(X_train_scale, y_train) * 100))
print('Test accuracy: %.2f%%' % (clf.score(X_test_scale, y_test) * 100))
print('AUC: %.2f%%' % (metrics.roc_auc_score(y_test,y_pred_prob) * 100))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_prob)


# In[409]:


coefs = pd.DataFrame(clf_gridsearch.coef_[0], index = X_train.columns, columns=['coefs'])
coefs


# #### Decision Tree

# In[410]:


# tuning hyperparameters
clf = DecisionTreeClassifier(criterion = 'entropy')

# Tune the hyperparameter
param_grid = {'max_depth': [3, 5, 10, 20],
             'min_samples_leaf': [3, 10, 50],
             'max_features': [3, 5, 7]}

# Grid Search
grid_search = GridSearchCV(clf, param_grid, cv=5, return_train_score=True, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

print('Grid Search Params')
print(grid_search.best_params_)

# apply to model to get scores
clf_gridsearch = grid_search.best_estimator_

scores = cross_validate(clf_gridsearch, X_train, y_train, cv=5, scoring = ['accuracy', 'roc_auc'], return_train_score=True)

print('Cross Validation Results')
print('Train accuracy CV: %.4f' % scores['train_accuracy'].mean())
print('Test accuracy CV: %.4f' % scores['test_accuracy'].mean())
print('Test AUC: %.4f' % scores['test_roc_auc'].mean())


# In[411]:


clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 20, max_features = 7, min_samples_leaf = 50)
clf.fit(X_train, y_train)

# apply model to test set
y_pred_prob = clf.predict_proba(X_test)[:,1]
y_pred = clf.predict(X_test)

print('Training accuracy: %.2f%%' % (clf.score(X_train, y_train) * 100))
print('Test accuracy: %.2f%%' % (clf.score(X_test, y_test) * 100))
print('AUC: %.2f%%' % (metrics.roc_auc_score(y_test,y_pred_prob) * 100))

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

# creat roc curve
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_prob)


# In[412]:


feature_importances = pd.DataFrame(clf.feature_importances_, index = X.columns, columns=['importance']).sort_values('importance', ascending=False)

plt.figure(figsize = [5,5])
plt.barh(feature_importances.index, feature_importances['importance'], color = '#7e57c2')
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance')
plt.show()


# In[413]:


from sklearn.tree import plot_tree

plt.figure(figsize=(12,12))
plot_tree(clf, feature_names=X.columns, class_names=['Renew', 'Churn'] 
   , filled = True, impurity = False, precision = 1, rounded = True, max_depth = 2, fontsize = 12)
plt.show()


# In[414]:


from IPython.display import SVG, display, Image
from graphviz import Source

graph = Source(export_graphviz(clf, out_file=None
   , feature_names=X.columns, class_names=['Renew', 'Churn'] 
   , filled = True, impurity = False, precision = 1, special_characters=True, rounded = True))
display(SVG(graph.pipe(format='svg')))

graph.render(filename='churn_tree')


# In[98]:


plt.figure(figsize = (6,3))

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_dt, tpr_dt, label = 'Decision Tree', color = '#7e57c2')
plt.plot(fpr_linear, tpr_linear, label = 'Linear Probability', color = '#009688')
plt.plot(fpr_log, tpr_log, label = 'Logistic Regression', color = '#c77025')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.legend()
plt.show()


# #### Make Predictions on current customers

# In[294]:


# apply model to predict set (current customers)
y_current_prob = clf.predict_proba(X_predict)[:,1]

pd.DataFrame(y_current_prob).describe()


# In[297]:


# put into table and save it
current_churnpred = pd.concat([df_model[df_model['churn_flag'] == 'current'][['subid']].reset_index(drop = True), 
           pd.DataFrame({'churn_prob': y_current_prob})], axis = 1)

current_churnpred.to_pickle('current_churnpred')
current_churnpred.head()


# ### Revenue Modeling

# In[257]:


discount = 0.1
acceptance = 0.2
price = 4.5

list_prices = [price, price*(1-discount), price*(1-discount), 0]


# In[191]:


simulations = 10

# for a specified number of simulatiosn and given threshold, calculate average number of people taking full price, offer, churn
def rev_simulation(threshold, simulations):
    c_fullprice = 0
    c_offer_renew = 0
    c_offer_churn = 0
    c_churn = 0
    
    for s in range(0,simulations):

        for i in range(0, len(y_pred_prob)):
            #determine if you want to give them the offer
            if y_pred_prob[i] > threshold:
                give_offer = 1
            else:
                give_offer = 0

            # determine if they will take up the offer
            if np.random.random() < acceptance:
                accept_offer = 1
            else:
                accept_offer = 0

            # determine what they do
            if give_offer * accept_offer > 0: # they take offer
                if y_test.values[i] == 0: #offer but they renew
                    c_offer_renew += 1
                else:
                    c_offer_churn += 1
            elif y_test.values[i] == 0: # they take full price
                c_fullprice += 1
            else: # they churn
                c_churn += 1
        
    return [c_fullprice/simulations, c_offer_renew/simulations, c_offer_churn/simulations, c_churn/simulations]


# In[258]:


# test which threshold leads to most revenue
# use simulation of 10 for ease of analysis

list_rev = []

for i in range(0,100):
    rev = np.multiply(rev_simulation(i/100,10), list_prices).sum()
    list_rev.append(rev)
    
plt.plot(range(0,100), list_rev)
plt.xlabel('Threshold %')
plt.show()


# Revenue is highest when you offer it to everyone. This makes sense because the churn rate is so high.

# In[263]:


# revenue for do nothing
rev_donothing = np.multiply(rev_simulation(1,1), list_prices).sum()
rev_donothing


# In[264]:


def print_revresults(sim): #input the results of the simulation
    revperprice = np.multiply(sim, list_prices)
    totalrev = revperprice.sum()
    revdiff = (totalrev - rev_donothing)/totalrev

    print('Take Up:')
    print(sim)
    print('Revenue Per Price:')
    print(revperprice)
    print('Total revenue:')
    print(totalrev)
    print('% change:')
    print(revdiff)


# In[266]:


# do nothing case
s_100 = rev_simulation(1,1)

print_revresults(s_100)


# In[267]:


# offer to everyone
s_0 = rev_simulation(0,10000)

print_revresults(s_0)


# In[268]:


# threshold 50%

#s_50 rev_simulation(0.5,10000)
print_revresults(s_50)


# In[269]:


# threshold 25%

#s_50 rev_simulation(0.25,10000)
print_revresults(s_25)

