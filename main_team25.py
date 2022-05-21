#!/usr/bin/env python
# coding: utf-8

# # Nuwe Schneider Electric Hackathon -- main
#     20220521, team 25: Kyla Juett, Leane Macachor, Alicja Mankowska

# ## Business Case
#     fbvbcvb

# ## Getting Started

# In[124]:


# import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas_profiling
from typing import List
from ipywidgets import HTML, Button, widgets
from pandas_profiling.report.presentation.core import Alerts

import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

import warnings
warnings.filterwarnings('ignore')


# In[162]:


# load the csv train datasets
df1 = pd.read_csv('/Users/kylajuett/projects/20220531 nuwe_se/datasets/train1.csv')#, index_col=0) 
df2 = pd.read_csv('../datasets/train2.csv', delimiter=';')


# In[163]:


# load the json train datasets
df3 = pd.read_json('http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/first') 
df4 = pd.read_json('http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/second') 
df5 = pd.read_json('http://schneiderapihack-env.eba-3ais9akk.us-east-2.elasticbeanstalk.com/third') 


# In[236]:


# load test csv 
test_x2 = pd.read_csv('/Users/kylajuett/projects/20220531 nuwe_se/datasets/test_x.csv')#, index_col=0) 
test_x2


# ##### import PDFs ??
#     Nuwe Carlos advice: PDFs are <1% of total data; if time at the end of the day (to get the extra points), do the import, if not, you don't really need it

# In[165]:


# !pip install tabula-py
# !pip3 install PyPDF2


# In[166]:


# import PyPDF2
# pdfFileObj = open('../datasets/pdfs-1.pdf', 'rb')

# pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
# print(pdfReader.numPages)
# pageObj = pdfReader.getPage(0)

# print(pageObj.extractText())
# pdfFileObj.close()


# In[167]:


# import tabula
# file_path = '/Users/kylajuett/projects/20220531 nuwe_se/datasets/pdfs-1.pdf'
# df = tabula.read_pdf(file_path, pages='all')
# df


# ## Functions
#     copied from imported files so it works elsewhere

# In[168]:


def C_metrics_train(model, X_train, y_train):
    scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'f1_macro': 'f1_macro'}
    
    scores = cross_validate(model, X_train, y_train, cv=10, scoring=scoring)
    ypredTrain = model.predict(X_train)
    Acc_train = scores['test_acc'].mean()
    Precision_train = scores['test_prec_macro'].mean()
    Recall_train = scores['test_rec_macro'].mean()
    F1_train = scores['test_f1_macro'].mean()
    conf_matrix_train = confusion_matrix(y_train, ypredTrain)
    from sklearn.metrics import classification_report
    statist_train = []
   
    list_metrics = [Acc_train, Precision_train, Recall_train, F1_train]
    statist_train.append(list_metrics)
    statist_train = pd.DataFrame(statist_train,columns = ['Accuracy', 'Precision', 'Recall', 'F1'], index = ['Train'])
    
    print('-----------------------------------------')
    print('TRAIN results')
    print('-----------------------------------------')
    print('Confusion Matrix \n', conf_matrix_train)
    return statist_train


def C_metrics_test(model, X_test, y_test):
    
    scoring = {'acc': 'accuracy',
           'prec_macro': 'precision_macro',
           'rec_macro': 'recall_macro',
           'f1_macro': 'f1_macro'}
    
    scores = cross_validate(model, X_test, y_test, cv=10, scoring=scoring)
    ypredtest = model.predict(X_test)
    report = classification_report(y_test, ypredtest,zero_division=0, output_dict=True)
    report = pd.DataFrame(report).T
    
    Acc_test = report.loc['accuracy', :].mean()  
    Rest_metrics = report.iloc[:-3,:]
    
    Precision_test = Rest_metrics.loc[:,'precision'].mean()
    Recall_test = Rest_metrics.loc[:,'recall'].mean()
    F1_test = Rest_metrics.loc[:,'f1-score'].mean()
    conf_matrix_test = confusion_matrix(y_test, ypredtest)
    
    statist_test = []
   
    list_metrics = [Acc_test, Precision_test, Recall_test, F1_test]
    statist_test.append(list_metrics)
    statist_test = pd.DataFrame(statist_test, columns = ['Accuracy', 'Precision', 'Recall', 'F1'], index = ['test'])
     
    print('-----------------------------------------')
    print('TEST results')
    print('-----------------------------------------')
    print('Confusion Matrix \n', conf_matrix_test)
    print(' Classification Report \n', Rest_metrics)
    return statist_test


def C_Allmetrics(model, X_train, y_train, X_test, y_test):
    
    stats_train = C_metrics_train(model, X_train, y_train)
    stats_test = C_metrics_test(model, X_test, y_test)
    final_metrics = pd.concat([stats_train, stats_test])
    print()
    print('++++++++ Summary of the Metrics ++++++++')
    print(final_metrics)
    return final_metrics


# In[169]:


def GetBasedModels():
    basedModels = []
    basedModels.append(('LR'   , LogisticRegression()))
    basedModels.append(('KNN'  , KNeighborsClassifier()))
    basedModels.append(('CART' , DecisionTreeClassifier()))
    basedModels.append(('SVM'  , SVC()))
    basedModels.append(('RF'   , RandomForestClassifier()))
    #basedModels.append(('ET'   , ExtraTreesClassifier())) 
    #basedModels.append(('LDA'  , LinearDiscriminantAnalysis()))
    #basedModels.append(('NB'   , GaussianNB()))
    #basedModels.append(('AB'   , AdaBoostClassifier()))
    #basedModels.append(('GBM'  , GradientBoostingClassifier()))
    return basedModels


def BasedModels(X_train, y_train, scoring, models):
    """
    BasedModels will return the evaluation metric ['AUC'] after performing
    a CV for each of the models
    input:
    X_train, y_train, scoring, models
    models = array containing the different models previously instantiated 
    
    output:
    names = names of the diff models tested
    results = results of the diff models
    """

    num_folds = 10
    scoring = scoring
    results = []
    names = []
    
    for name, model in models:
        cv_results = cross_val_score(model, X_train,
                                     y_train, cv=num_folds, scoring=scoring)
        results.append(cv_results.mean())
        names.append(name)
        msg = "%s: %s = %f (std = %f)" % (name, scoring,
                                                cv_results.mean(), 
                                                cv_results.std())
        print(msg)
    scoreDataFrame = pd.DataFrame({'Model':names, 'Score': results})    
    return scoreDataFrame


def GetScaledModel(nameOfScaler):
    """
    arg:
    nameOfScaler = 'standard' (standardize),  'minmax', or 'robustscaler'
    """
    
    if nameOfScaler == 'standard':
        scaler = StandardScaler()
    elif nameOfScaler =='minmax':
        scaler = MinMaxScaler()
    elif nameOfScaler == 'robustscaler':
        scaler = RobustScaler()

    pipelines = []
    pipelines.append((nameOfScaler+'LR', 
                      Pipeline([('Scaler', scaler),
                                ('LR', LogisticRegression())])))
    
    pipelines.append((nameOfScaler+'KNN', 
                      Pipeline([('Scaler', scaler),('KNN', 
                                                   KNeighborsClassifier())])))
    pipelines.append((nameOfScaler+'CART', 
                      Pipeline([('Scaler', scaler),
                                ('CART', DecisionTreeClassifier())])))
    pipelines.append((nameOfScaler+'SVM',
                      Pipeline([('Scaler', scaler),
                                ('SVM', SVC(kernel = 'rbf'))])))
    pipelines.append((nameOfScaler+'RF', 
                      Pipeline([('Scaler', scaler),
                                ('RF', RandomForestClassifier())])))
    
    #pipelines.append((nameOfScaler+'ET'  , Pipeline([('Scaler', scaler),('ET'  , ExtraTreesClassifier())])  ))
    #pipelines.append((nameOfScaler+'LDA' , Pipeline([('Scaler', scaler),('LDA' , LinearDiscriminantAnalysis())])))
    #pipelines.append((nameOfScaler+'NB'  , Pipeline([('Scaler', scaler),('NB'  , GaussianNB())])))
    #pipelines.append((nameOfScaler+'AB'  , Pipeline([('Scaler', scaler),('AB'  , AdaBoostClassifier())])  ))
    #pipelines.append((nameOfScaler+'GBM' , Pipeline([('Scaler', scaler),('GMB' , GradientBoostingClassifier())])  ))

    return pipelines 


# In[170]:


def matchingElements(dictionary, searchString):
    return {key:val for key,val in dictionary.items() if any(searchString in s for s in key)}


def OutLiersBox(df, nameOfFeature):
    '''args: df, nameOfFeature'''
    trace0 = go.Box(
        y = df[nameOfFeature],
        name = "All Points", jitter = 0.3, pointpos = -1.8, boxpoints = 'all', 
        marker = dict(color = 'rgb(7,40,89)'), line = dict(color = 'rgb(7,40,89)')
    )
    trace1 = go.Box(
        y = df[nameOfFeature],
        name = "Suspected Outliers",
        boxpoints = 'suspectedoutliers', # define the suspected Outliers
        marker = dict(color = 'rgba(219, 64, 82, 0.6)',
            #outliercolor = 'rgba(219, 64, 82, 0.6)',
            line = dict(outlierwidth = 2)),
        line = dict(color = 'rgb(8,81,156)')
    )
    data = [trace0, trace1]
    layout = go.Layout(
        title = "{} Outliers".format(nameOfFeature)
    )
    fig = go.Figure(data=data,layout=layout)
    fig.show()
    #fig.write_html("{}_file.html".format(nameOfFeature))

def allOutLiersBox():
    for item in range(0, len(df.columns)):
        #print(item)
        OutLiersBox(df, df.columns[item])
        print('----------------------------------------------')


# ## EDA

# In[171]:


print(df1.duplicated().sum()) # none
print(df1.isna().sum()) # none
print(df1.info()) # so clean! <3
# df1.head(-10)


# In[172]:


print(df2.duplicated().sum()) # 3775
print(df2.isna().sum()) # none
print(df2.info()) # beautiful
# df2.head(-10)


# In[173]:


print(df3.duplicated().sum()) # none
print(df3.isna().sum()) # none
print(df3.info()) # so clean! <3
# df3.head(-10)


# In[174]:


dfcsv = pd.concat([df1,df2])
dfjson = pd.concat([df3,df4,df5])


# In[175]:


print(dfcsv.columns)
dfcsv


# In[176]:


print(dfjson.columns)
dfjson


# ### Concatenate csv & json

# In[177]:


df = pd.concat([dfcsv,dfjson])
# df


# In[178]:


test_x['Type'] = 'Test'
df['Type'] = 'Train'
df = pd.concat([df, test_x])
df.shape


# In[179]:


print(df.targetRelease.nunique())
print(df.CONTINENT.nunique())


# In[180]:


df = df.drop(['', 'EPRTRAnnexIMainActivityCode', 'EPRTRSectorCode', 'targetRelease', 'CONTINENT', 'REPORTER NAME'], axis=1)

df = df.drop_duplicates()
# df


# In[181]:


df = df.reset_index().drop('index', axis=1)


# In[182]:


# encoding pollutants
pollutants_dict = {'Nitrogen oxides (NOX)': 0, 'Carbon dioxide (CO2)': 1, 'Methane (CH4)': 2}

df.pollutant = df.pollutant.replace(pollutants_dict)
df


# In[183]:


df_copy = df.copy()


# In[225]:


df_copy


# In[184]:


# encoding mainActivityLabel 
main_act = dict(zip(dfjson.EPRTRAnnexIMainActivityLabel, dfjson.EPRTRAnnexIMainActivityCode))
df.EPRTRAnnexIMainActivityLabel = df.EPRTRAnnexIMainActivityLabel.replace(main_act)
# df


# In[185]:


# encoding sector name 
sector_dict = {'Energy sector': 1, 'Waste and wastewater management': 5, 'Mineral industry': 3, 'Chemical industry': 4, 
              'Paper and wood production and processing': 6, 'Production and processing of metals': 2,
              'Intensive livestock production and aquaculture': 7, 'Animal and vegetable products from the food and beverage sector': 8, 
              'Other activities': 9}
df.eprtrSectorName = df.eprtrSectorName.replace(sector_dict).astype('int64')
df


# In[186]:


# encoding country name 
lb_enc = LabelEncoder()
df['countryName_code'] = lb_enc.fit_transform(df['countryName']) 


# In[187]:


# numerical encoding mainActivityCode
# df.EPRTRAnnexIMainActivityCode.unique().tolist()
main_code = {'3(c)(i)':'3,3,1',
 '3(c)':'3,3,0',
 '5(d)':'5,4,0',
 '1(c)':'1,3,0',
 '5(f)':'5,6,0',
 '1(a)':'1,1,0',
 '3(e)':'3,5,0',
 '6(b)':'6,2,0',
 '4(a)(i)':'4,1,1',
 '5(b)':'5,2,0',
 '2(b)':'2,2,0',
 '3(c)(iii)':'3,3,3',
 '4(a)(viii)':'4,1,8',
 '7(a)(iii)':'7,1,3',
 '3(a)':'3,1,0',
 '6(a)':'6,1,0',
 '2(d)':'2,4,0',
 '5(a)':'5,1,0',
 '8(b)(ii)':'8,2,2',
 '4(a)':'4,1,0',
 '5(c)':'5,3,0',
 '7(a)(ii)':'7,1,2',
 '4(b)':'4,2,0',
 '7(a)':'7,1,0',
 '2(a)':'2,1,0',
 '4(b)(i)':'4,2,1',
 '3(g)':'3,7,0',
 '4(a)(ii)':'4,1,2',
 '2(e)(i)':'2,5,1',
 '4(c)':'4,3,0',
 '3(b)':'3,2,0',
 '2(f)':'2,6,0',
 '2(e)(ii)':'2,5,2',
 '8(b)':'8,2,0',
 '8(c)':'8,3,0',
 '4(e)':'4,5,0',
 '2(e)':'2,5,0',
 '1(d)':'1,4,0',
 '4(b)(v)':'4,2,5',
 '8(b)(i)':'8,2,1',
 '7(a)(i)':'7,1,1',
 '5(e)':'5,5,0',
 '3(f)':'3,6,0',
 '9(d)':'9,4,0',
 '9(c)':'9,3,0',
 '4(b)(iv)':'4,2,4',
 '1(b)':'1,2,0',
 '2(c)(i)':'2,3,1',
 '1(e)':'1,5,0',
 '2(c)(iii)':'2,3,3',
 '4(a)(iv)':'4,1,4',
 '3(c)(ii)':'3,3,3',
 '4(b)(iii)':'4,2,3',
 '4(a)(xi)':'4,1,11',
 '4(a)(x)':'4,1,10',
 '4(b)(ii)':'4,2,2',
 '2(c)':'2,3,0',
 '8(a)':'8,1,0',
 '6(c)':'6,3,0',
 '9(a)':'9,1,0',
 '4(a)(vi)':'4,1,6',
 '4(a)(ix)':'4,1,9',
 '1(f)':'1,6,0',
 '4(d)':'4,2,0',
 '5(g)':'5,7,0',
 '4(a)(iii)':'4,1,3',
 '2(c)(ii)':'2,3,2',
 '9(e)':'9,5,0',
 '4(a)(v)':'4,1,5',
 '4(f)':'4,6,0'}

df.EPRTRAnnexIMainActivityLabel = df.EPRTRAnnexIMainActivityLabel.replace(main_code)
df


# In[188]:


df[['EPRTRAnnexIMainActivityCode_0', 'EPRTRAnnexIMainActivityCode_1', 'EPRTRAnnexIMainActivityCode_2']] = df.EPRTRAnnexIMainActivityLabel.str.split(',', expand=True)
df


# In[189]:


df['facilityName_code'] = lb_enc.fit_transform(df['facilityName']).astype('int64')
df['CITY ID_code'] = lb_enc.fit_transform(df['CITY ID']).astype('int64') 
# df


# In[190]:


df = df[~df.EPRTRAnnexIMainActivityLabel.isin(['Chemical installations for the production on an industrial scale of basic organic chemicals: Organometallic compounds'])]
# df


# In[191]:


dftest = df.loc[df['Type'].isin(['Test'])]
dftest = dftest.reset_index().drop('index', axis=1)
dftest


# In[202]:


df = df.loc[df['Type'].isin(['Train'])]
df = df.drop(['test_index'], axis=1)
df


# In[203]:


g = sns.pairplot(df, hue="pollutant", palette="plasma")


# In[250]:


df.hist(edgecolor='white', figsize=(20, 20))


# In[204]:


prof = df.profile_report(sort=None)
prof.to_file(output_file='profile_report_pollution5.html') # 
prof
# important correlations to target (pollutant): max_wind_speed, avg_wind_speed, min_wind_speed; max_temp, avg_temp, min_temp; eprtrSectorName, fog 


# ## Feature Importance

# In[194]:


# define the target
X = df[['EPRTRAnnexIMainActivityCode_1', 'eprtrSectorName', 'facilityName_code', 
        'EPRTRAnnexIMainActivityCode_0', 'max_wind_speed', 'min_wind_speed', 'avg_wind_speed', 
        'min_temp', 'max_temp', 'CITY ID_code', 'avg_temp', 'DAY', 'countryName_code', 
        'reportingYear', 'MONTH', 'DAY WITH FOGS', 'EPRTRAnnexIMainActivityCode_2']] 
y = df['pollutant']


# In[195]:


# instantiate & fit the RF Classifier model
clf = RandomForestClassifier(random_state=37)
clf.fit(X,y)


# In[196]:


feature_importance = clf.feature_importances_
feature_importance


# In[197]:


# make importances relative to max importance 
feature_importance = 100.0 * (feature_importance / feature_importance.max())
feature_importance


# In[198]:


importance_ = pd.DataFrame(feature_importance,
                           index= X.columns)

importance_ = importance_.reset_index()
importance_.columns = ['Column_names', 'Importance_%']

importance_ = importance_.sort_values(by= 'Importance_%', ascending=False)
importance_


# In[199]:


px.bar(importance_, x= 'Importance_%', y= 'Column_names', color= 'Column_names',orientation='h')


# In[98]:


# importance_['Column_names']


# # Pipeline! Classification Models

# In[217]:


# define baseline models
models = GetBasedModels()
models


# In[218]:


scoring = 'f1_macro'


# In[219]:


Base_model = BasedModels(X_train, y_train, 'roc_auc', models) 


# In[ ]:


Base_model = BasedModels(X_train, y_train, 'accuracy', models)


# In[220]:


# this is the one that matters here
Base_model = BasedModels(X_train, y_train, 'f1_macro', models) 


# In[ ]:


MetricsClas(models,X_train, y_train, X_test, y_test)
# and the winner is... Random Forest! 


# #### Winning Model: Random Forest
# 

# In[ ]:





# # Modeling: Random Forest Classifier

# In[206]:


# Train-Test Split within Train DF (target initialization commented out because we ran it above)
# X = df[['EPRTRAnnexIMainActivityCode_1', 'eprtrSectorName', 'facilityName_code', 
#         'EPRTRAnnexIMainActivityCode_0', 'max_wind_speed', 'min_wind_speed', 'avg_wind_speed', 
#         'min_temp', 'max_temp', 'CITY ID_code', 'avg_temp', 'DAY', 'countryName_code', 
#         'reportingYear', 'MONTH', 'DAY WITH FOGS', 'EPRTRAnnexIMainActivityCode_2']] 
# y = df['pollutant']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=37)    


# In[207]:


rand_forest = RandomForestClassifier(n_estimators=500, random_state=37)
rand_forest.fit(X_train, y_train)


# In[ ]:


y_pred_RF = rand_forest.predict(X_test)


# ### Evaluation of the Model

# In[208]:


print(classification_report(y_test, y_pred_RF))


# In[209]:


C_Allmetrics(rand_forest, X_train, y_train, X_test, y_test)


# ### RF Hypertuning

# In[210]:


param_grid = {'max_depth':[5, 50, 500], 
              'n_estimators':[50, 500, 1000],
              'min_samples_split':[4, 5, 6]}

rand_frst_clf = RandomForestClassifier(random_state=37)

grid_rand_forest = GridSearchCV(rand_frst_clf, param_grid, scoring="f1_macro", 
                                n_jobs= -1, cv=3)


# In[211]:


grid_rand_forest.fit(X_train, y_train)


# ### Best Estimator

# In[212]:


grid_rand_forest.best_estimator_

# RandomForestClassifier(max_depth=500, min_samples_split=4, n_estimators=500, random_state=37)


# In[213]:


rand_forest = RandomForestClassifier(max_depth=50, min_samples_split=4,
                    n_estimators=1400,
                    random_state=37)

rand_forest.fit(X_train, y_train)


# ### Re-Evaluation of the Model

# In[214]:


C_Allmetrics(rand_forest,X_train, y_train, X_test, y_test)


# In[ ]:





# # Test It
#     with the real Test df
#         dun-dun-dunnn

# In[221]:


print(test_x.duplicated().sum()) # none
print(test_x.isna().sum())
print(test_x.info()) 
test_x.head(-10)


# In[228]:


# define target / test-train split 
df_test = dftest[['eprtrSectorName', 'reportingYear', 'MONTH', 'DAY', 'max_wind_speed',
       'avg_wind_speed', 'min_wind_speed', 'max_temp', 'avg_temp', 'min_temp',
       'DAY WITH FOGS', 'EPRTRAnnexIMainActivityCode_0',
       'EPRTRAnnexIMainActivityCode_1', 'EPRTRAnnexIMainActivityCode_2',
       'countryName_code', 'facilityName_code', 'CITY ID_code']]
y = df['pollutant']


# In[230]:


# make (& view) predictions
y_pred = clf.predict(df_test)

print(y_pred.shape)
y_pred


# In[231]:


C_Allmetrics(clf_rfc, X_train, y_train, df_test, y_pred)


# In[237]:


df_test


# In[238]:


# save predictions to a new column in test_x DF (renamed, as above)
df_test['pollutant_pred'] = y_pred
df_test


# In[240]:


print(df_test['pollutant_pred'].value_counts())
df_test['pollutant_pred'].value_counts(normalize=True)


# In[241]:


print(df['pollutant'].value_counts())
df['pollutant'].value_counts(normalize=True)


# In[246]:


# Q: how close are the predictions (to each other)?  A: wow, that is a terrible model, OMG.
print('df_test (AKA test_x) actual predictions \n', df_test.pollutant_pred.value_counts())

orig_perc = df.pollutant.value_counts(normalize=True)
pred_perc = df_test.pollutant_pred.value_counts(normalize=True)

print('\n difference, in % \n', (orig_perc - pred_perc)*100)


# In[ ]:





# In[248]:


# wowww, that's a big change! 
g2 = sns.pairplot(df_test, hue="pollutant_pred", palette="plasma")


# In[249]:


df_test.hist(edgecolor='white', figsize=(20, 20))


# In[ ]:





# # Conclusion
#     So we have a TERRIBLE test prediction outcome, despite the fairly good train metrics (63-64% F1, 61% accuracy).
#     This might explain what happened here with our model: "F1 is a quick way to tell whether the classifier is actually good at identifying members of a class, or if it is finding shortcuts (e.g., just identifying everything as a member of a large* class)."
#         Source:  https://medium.com/analytics-vidhya/evaluating-a-random-forest-model-9d165595ad56#:~:text=F1%20score%20is%20a%20little,low%2C%20F1%20will%20be%20low.
# 
#     Our next challenge: figure out why this happened!

# In[ ]:





# # "This is the End"
#     - The Doors
#         https://www.youtube.com/watch?v=VScSEXRwUqQ

# In[244]:


results_team25 = df_test.pollutant_pred

results_team25 = pd.DataFrame(results_team25)
results_team25.to_csv('/Users/kylajuett/projects/20220531 nuwe_se/datasets/results_team25.csv')
# results_team25


# In[252]:


# results_team25.to_csv('/Users/kylajuett/projects/20220531 nuwe_se/datasets/results_team25.csv')
results_team25.to_json('/Users/kylajuett/projects/20220531 nuwe_se/datasets/results_team25.json')
df.to_csv('/Users/kylajuett/projects/20220531 nuwe_se/datasets/df_train.csv') 
df_test.to_csv('/Users/kylajuett/projects/20220531 nuwe_se/datasets/df_test__test_x.csv') 


# In[ ]:




