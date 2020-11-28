#%%
import sys
sys.path.append("../")
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import churn_analysis as utils
import basic_feature_analysis_pi as bsf
import course_utils as bd
import os

label = 'Underdog Win'
label_ind = 10 ## After you drop the columns
cols_drop = ['unique_game_id', 'DATE', 'Home Team', 'Away Team', 'Away Underdog']

main_path = '..\..\scraping\merging\cleaned_dfs_11-23\\all_rolling_windows\\'
filename = '\\2015_stats_n10.csv'
years = list(range(2009,2019)) #testing years for now

# %%
train_df = bsf.construct_df(years, 50, main_path)  ## arbitrary 20
#train_df = train_df.drop(columns=train_df.columns.values[65:]) ## lets drop the playstyle columns # idt 65 is right


cols = train_df.drop(label, 1).columns.values
rf_clf = RandomForestClassifier(criterion='entropy', class_weight='balanced_subsample')
rf_clf = rf_clf.fit(train_df.drop(label, 1), train_df[label])
clf_def = DecisionTreeClassifier(criterion='entropy', min_samples_leaf = 20)
clf_def = clf_def.fit(train_df.drop(label, 1), train_df[label])



rf_fi = rf_clf.feature_importances_
dt_fi = rf_clf.feature_importances_

## top 30
num_top = 63
top_fis_rf = sorted(list(zip(rf_fi, cols)), key = lambda x: x[0])[-num_top:]
bottom_fis_rf = sorted(list(zip(rf_fi, cols)), key = lambda x: x[0])[:len(rf_fi) - num_top]

#top_fis_dt = sorted(list(zip(dt_fi, cols)), key = lambda x: x[0])[-num_top:]
#bottom_fis_dt = sorted(list(zip(dt_fi, cols)), key = lambda x: x[0])[:len(dt_fi) - num_top]



# fig, ax = plt.subplots()
fig = plt.figure(figsize = (12, 6))
ax = plt.subplot(111)


width=0.3


# print(len([x[0] for x in top_fis_rf]))
# print((np.arange(num_top)+width).shape)


ax.bar(np.arange(num_top)+width, [x[0] for x in top_fis_rf], width, color='r', label='RF')
#ax.bar(np.arange(num_top)+width, [x[0] for x in top_fis_dt], width, color='b', label='DT')

ax.set_xticks(np.arange(num_top))
ax.set_xticklabels([x[1] for x in top_fis_rf], rotation=90)
plt.title('Feature Importance from RF')
ax.set_ylabel('Normalized Gini Importance')
plt.legend(loc=1)
# %%
## Now lets see how a random forest does with minimal tuning, no other playstyle features
test_years = [2019, 2020]
test_df = bsf.construct_df(test_years, 50, main_path)  ## arbitrary 50
#test_df = test_df.drop(columns=test_df.columns.values[65:]) ## lets drop the playstyle columns

rf_pred_proba = rf_clf.predict_proba(test_df.drop(label,1))
rf_pred = rf_clf.predict(test_df.drop(label,1))
bd.plotAUC(test_df[label], rf_pred_proba[:,1], 'RF')
plt.show()


# %%
plt.close()
print(len(rf_pred), np.sum(rf_pred))
#plt.plot(np.arange(len(rf_pred_proba)), rf_pred_proba[:,1])

threshold_offset = 0.3
rf_pred_proba2 = rf_pred_proba[:,1] + threshold_offset
rf_pred_proba2 = np.around(rf_pred_proba2, 0)


bd.plotAUC(test_df[label], rf_pred_proba2, 'RF')
# %%
## to do: upsample, feature engineering, etc

###_________________________________ BUILDING HYPER PARAM PIPELINE TEST

#binary_vars = utils.getDfSummary(test_df)
#binary_vars2 = binary_vars[binary_vars['distinct'] == 2]
# From the above code we know that we only have 2 binary vars (one is the label; the other
# is no necessary (away underdog))

def create_train_test(train_beg, train_end, test_beg, test_end, n, preprocess=True):
    '''
    Enter beginning and end years as well as n and get train and test df back
    '''
    train_years = list(np.arange(train_beg, train_end+1))
    test_years = list(np.arange(test_beg, test_end+1))

    train_df = bsf.construct_df(train_years, n, main_path)
    test_df = bsf.construct_df(test_years, n, main_path)

    X_train = train_df.drop(label, 1)
    y_train = train_df[label]
    X_test = test_df.drop(label, 1)
    y_test = test_df[label]

    '''if preprocess:
        X_train, X_test = preprocess_df(X_train, X_test)'''

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = create_train_test(2009, 2018, 2019, 2020, 50)

def upsample(X_train, y_train):
    '''
    Given a training set, upsample the minority class 
    '''
    upsampled_df = []

    return upsampled_df

def make_pipeline_rf(std=True, pca=False, poly=False):
    ## Ignoring poly and pca for now
    ## Assumes standard scalar
    
    steps = [('scalar', sk.preprocessing.StandardScaler()), ('rf', RandomForestClassifier())]
    pipeline = Pipeline(steps) 
    
    return pipeline


# without pipeline
param_grid_lr = {'C':[10**i for i in range(-3, 3)], 'penalty':['l1', 'l2']}


# with pipeline
param_grid_rf = dict(rf__n_estimators = [10**i for i in range(2, 3)],
                  rf__criterion = ['gini', 'entropy'],
                  rf__class_weight = [None, 'balanced', 'balanced_subsample'],
                  rf__max_features = ['auto', 'sqrt', 'log2'],
                  rf__min_samples_split = [2, 10])



param_grid_gbt = dict(pca__n_components = [6, 10, 16, 20, 25, 30, 35, 50],
                     gbt__n_estimators = [10**i for i in range(2, 3)],
                     gbt__max_depth = [3,4,5],
                     gbt__max_features = ['sqrt', 'log2', 5],
                     gbt__min_samples_split = [10, 100, 200, 300],
                     gbt__min_samples_leaf = [10, 100, 200, 300])


# def model_bakeoff():
#     best_dataset = []

#     for datagroup in thatlist:
    
#         pipeline_rf = make_pipeline_rf()
#         pipeline_gbt = make_pipeline_gbt()

#         ## Choose x train, and y train from the datagroup
#         model_gscv = hyper_param_search(X_train, y_train, pipeline, param_grid_rf, 10)


def hyper_param_search(X_train, y_train, pipeline, param_grid, num_folds):
    '''
    Assumes X_train, X_test, y_train, y_test are already made and variables accessible to the function.
    Even though random forests have a built in CV of sorts with OOB, we will still cross validate. To minimize
    time wrt CV just make kfolds = 2
    '''
    kfolds = KFold(n_splits = num_folds)
    model_grid_search_scaler = GridSearchCV(pipeline, param_grid=param_grid, cv = kfolds, scoring = 'roc_auc')
    model_grid_search_scaler.fit(X_train, y_train)

    # thinking about returning or writing to some file the best model
    return model_grid_search_scaler




#%%
pipeline = make_pipeline_rf()

model_gscv = hyper_param_search(X_train, y_train, pipeline, param_grid_rf, 10)
# %%
## test AUC for the best estimator
best_predictions = model_gscv.best_estimator_.predict(X_test)
best_proba = model_gscv.best_estimator_.predict_proba(X_test)

bd.plotAUC(y_test, best_proba[:,1], 'RF')
plt.show()

# %%
plt.close()
rf2 = RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy', min_samples_split=10)
rf2 = rf2.fit(X_train, y_train)
rf2_predict = rf2.predict_proba(X_test)
bd.plotAUC(y_test, rf2_predict[:,1], 'RF')
# %%
## next: try gradient boosted trees... i have a feeling this wont get us there tho 
# 
plt.close()
gbt = GradientBoostingClassifier()
gbt_f = gbt.fit(X_train, y_train)
gbt_predict = gbt_f.predict_proba(X_test)
bd.plotAUC(y_test, gbt_predict[:,1], 'GBT')

#%%





# next2: run through multiple models, multiple pipelines, different tests, different test data, subset selection, 
# different scoring metrics other than roc_auc
## etc and just see which is best