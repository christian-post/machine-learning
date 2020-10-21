# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:11:06 2019

@author: Christian Post


data from
http://odds.cs.stonybrook.edu/mammography-dataset/

Dataset Information

The original Mammography (Woods et al., 1993) data set was made available by the 
courtesy of Aleksandar Lazarevic. This dataset is publicly available in openML. 
It has 11,183 samples with 260 calcifications. If we look at predictive accuracy
as a measure of goodness of the classifier for this case, the default accuracy 
would be 97.68% when every sample is labeled non-calcification. But, it is
desirable for the classifier to predict most of the calcifications correctly. 
For outlier detection, the minority class of calcification is considered as 
outlier class and the non-calcification class as inliers.  

"""

import scipy.io
import pandas as pd
import random
from itertools import combinations, product
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from score import plot_confusion_matrix, matthews_corr, matthews_corr2


random.seed(13)
pd.options.mode.chained_assignment = None


mat = scipy.io.loadmat('mammography.mat')
data_X = pd.DataFrame(mat['X']) 
data_y = pd.DataFrame(mat['y'])

data = pd.concat([data_X, data_y], axis=1)

features = ['attr0', 'attr1', 'attr2', 'attr3', 'attr4', 'attr5']
data.columns = features + ['label']


# plausibilisation
data = data.loc[data['attr3'] > -0.85955]

# normalisation
data[features] = (data[features]-data[features].min())/(data[features].max()-data[features].min())



def print_predictions(cm, name, printing=True):
    tn, fp, fn, tp = cm.ravel()
    
    if (tp + fn) == 0:
        sen = np.nan
    else:
        sen = round(tp / (tp + fn), 2)
    
    if (tn + fp) == 0:
        spe = np.nan
    else:
        spe = round(tn / (tn + fp), 2)
    
    if (tp + fp) == 0:
        ppv = np.nan
    else:
        ppv = round(tp / (tp + fp), 2)
    
    if (tn + fn) == 0:
        npv = np.nan
    else:
        npv = round(tn / (tn + fn), 2)
    
    if printing:
        print(f'\n {name} Predictions:')
        #print(cm)
        print(f'Sensitivity: {sen}')
        print(f'Specificity: {spe}')
        print(f'PPV: {ppv}')
        print(f'NPV: {npv}')
    
    return sen, spe, ppv, npv


train, test = train_test_split(data, test_size=.2, random_state=13)
train_normal = train[train['label']==0]
train_outliers = train[train['label']==1]
outlier_prop = len(train_outliers) / len(train_normal)

test_normal = test[test['label']==0]
test_outliers = test[test['label']==1]
outlier_prop_test = len(test_outliers) / len(test_normal)

data_pre = len(data[data["label"]==1]) / len(data)
train_pre = len(train[train["label"]==1]) / len(train)
test_pre = len(test[test["label"]==1]) / len(test)

print(f'\nPrevalence in complete data: {len(data[data["label"]==1])}/{len(data)} ({round(data_pre, 2)} %)')
print(f'Prevalence in train data: {len(train[train["label"]==1])}/{len(train)} ({round(train_pre, 2)} %)')
print(f'Prevalence in test data: {len(test[test["label"]==1])}/{len(test)} ({round(test_pre, 2)} %)')


# split into training and validation data

train, validation = train_test_split(train, test_size=.2, random_state=13)
train_normal = train[train['label']==0]
train_outliers = train[train['label']==1]
outlier_prop = len(train_outliers) / len(train_normal)


# -- build classifiers --

clf_classes = {
        'svm': SVC,
        'rf': RandomForestClassifier,
        'knn': KNeighborsClassifier,
        'lr': LogisticRegression,
        'dt': DecisionTreeClassifier,
        'gnb': GaussianNB,
        'ada': AdaBoostClassifier,
        'qda': QuadraticDiscriminantAnalysis,
        'et': ExtraTreesClassifier,
        'gb': GradientBoostingClassifier
        }

clf_params = {
        'svm': {
                'gamma': 'auto',
                'kernel': 'rbf',
                'class_weight': 'balanced',
                'probability': True,
                'random_state': 13
                },
        'rf': {
                'n_estimators': 500,
                'max_depth': 5,
                'class_weight': 'balanced',
                'max_features': 'auto',
                'random_state': 13
                },
        'lr': {
                'class_weight': 'balanced',
                'random_state': 13,
                'solver': 'lbfgs'
                },
        'dt': {
                'class_weight': 'balanced',
                'max_depth': 4,
                'random_state': 13
                },
        'et': {
                'n_estimators': 500,
                'max_depth': 4,
                'class_weight': 'balanced',
                'random_state': 13
                },
        'gb': {
                'n_estimators': 500,
                'max_depth': 4,
                'random_state': 13
                },
        }
        
# specify the base classifier for adaBoost
clf_params['ada'] = {
                'base_estimator': DecisionTreeClassifier(**clf_params['dt']),
                'n_estimators': 500,
                'random_state': 13,
                }

classifiers = {}
for key, model in clf_classes.items():
    if key in clf_params.keys():
        classifiers[key] = model(**clf_params[key])
    else:
        classifiers[key] = model()


clf_names = {
        'svm': 'Support Vector Machine',
        'rf': 'Random Forest',
        'knn': 'K-nearest Neighbors',
        'lr': 'Logistic Regression',
        'dt': 'Decision Tree',
        'gnb': 'GaussianNaiveBayes',
        'ada': 'AdaBoost',
        'mlpc': 'Multilayer Perceptron',
        'qda': 'Quadratic Discriminant Analysis',
        'et': 'ExtraTrees Classifier',
        'gb': 'Gradient Boosting'
        }


# data base for classifier scores
scores = {'name': [], 'sen_validation': [], 'sen_test': [], 
          'spe_validation': [], 'spe_test': [], 
          'ppv_validation': [], 'ppv_test': [], 
          'npv_validation': [], 'npv_test': [], 
          'matthews_validation': [], 'matthews_test': [],}

conf_matrices = {}


# --fit and test --
predictions = {}

# fit to train and test on validation
for name, clf in classifiers.items():
    conf_matrices[name] = {}
    
    scores['name'].append(clf_names[name])
    clf.fit(train[features], train['label'])
    y_pred = clf.predict(validation[features])
    cm = confusion_matrix(validation['label'], y_pred)
    pvalues = print_predictions(cm, f'{clf_names[name]} on Validation', False)
    predictions[f'{name}_validation'] = [y_pred, clf.predict_proba(validation[features])]
    
    scores['sen_validation'].append(pvalues[0])
    scores['spe_validation'].append(pvalues[1])
    scores['ppv_validation'].append(pvalues[2])
    scores['npv_validation'].append(pvalues[3])
    scores['matthews_validation'].append(matthews_corr(cm))
    

# filter only the predicted positives from the data
for key, value in predictions.items():
    validation[f'pred_{key}'] = value[0]
    validation[f'proba_{key}'] = [v[1] for v in value[1]]
    



# fit the classifiers to train data and test on test data
predictions = {}
for name, clf in classifiers.items():
    clf.fit(train[features], train['label'])
    y_pred = clf.predict(test[features])
    cm = confusion_matrix(test['label'], y_pred)
    conf_matrices[name]['test'] = cm.ravel()
    pvalues = print_predictions(cm, f'{clf_names[name]} on test set')
    predictions[f'{name}'] = [y_pred, clf.predict_proba(test[features])]
    
    plot_confusion_matrix(cm, [0, 1], False, f'{clf_names[name]} on Test set')

    scores['sen_test'].append(pvalues[0])
    scores['spe_test'].append(pvalues[1])
    scores['ppv_test'].append(pvalues[2])
    scores['npv_test'].append(pvalues[3])
    scores['matthews_test'].append(matthews_corr(cm))
    

# filter only the predicted positives from the data
for key, value in predictions.items():
    test[f'pred_{key}'] = value[0]
    test[f'proba_{key}'] = [v[1] for v in value[1]]



scores_df = pd.DataFrame(scores)

scores_df['f1_validation'] = 2 / (1 / scores_df['sen_validation'] 
                                  + 1 / scores_df['ppv_validation'])
scores_df['f1_test'] = 2 / (1 / scores_df['sen_test'] 
                                  + 1 / scores_df['ppv_test'])

scores_df = scores_df.sort_values(by='matthews_test', ascending=False)

scores_df.to_csv('mammo_classifiers.csv', sep=';', decimal=',',
                 index=False)






