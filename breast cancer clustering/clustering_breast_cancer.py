# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 14:20:27 2019

@author: Christian Post
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

# Breast cancer dataset description
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
cancer = load_breast_cancer()

features = cancer.feature_names

cancer_df = pd.DataFrame(data=cancer.data, columns=features)
cancer_df['target'] = cancer.target


# scale the data for KNN.
scaler = StandardScaler()
scaler.fit(cancer_df[features])
cancer_df[features] = scaler.transform(cancer_df[features])

# Create decision tree classifer object for feature selection
clf_params = {
        'random_state': 1,
        'n_jobs': -1,
        'n_estimators': 200,
        'max_depth': None
        }
clf = RandomForestClassifier(**clf_params)
# Train the model
clf.fit(cancer_df[features], cancer_df['target'])
# Calculate feature importances
importances = clf.feature_importances_
# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]
# Rearrange feature names so they match the sorted feature importances
names = [features[i] for i in indices]

# Create plot
plt.figure(figsize=(12, 6))
# Create plot title
plt.title("Feature Importance")
# Add bars
plt.barh(range(cancer_df[features].shape[1])[::-1], importances[indices])
# Add feature names as x-axis labels
plt.yticks(range(cancer_df[features].shape[1])[::-1], names)
plt.show()


# K Means
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

X = df[names]
# select 3 best features
#X = df[names[:3]]
y = df['target']

# Number of clusters
kmeans = KMeans(n_clusters=2)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
C = kmeans.cluster_centers_

plt.rcParams['figure.figsize'] = (12, 8)
fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(azim=60, elev=60)
ax.set_xlabel(names[0])
ax.set_ylabel(names[2])
ax.set_zlabel(names[3])
ax.scatter(X[names[0]], X[names[1]], X[names[2]], s=36, c=cancer.target)
ax.scatter(C[:, 0], C[:, 1], C[:, 2], marker='*', c='#050505', s=1000)
plt.savefig('Breast_cancer_clustering.png', bbox_inches='tight')
plt.show()


# KNN with 10 most important features
accuracies = []
for i in range(100):
    KNN = KNeighborsClassifier()
    X_train, X_valid, y_train, y_valid = train_test_split(cancer_df[names[:10]], 
                                                          cancer_df['target'],
                                                          test_size=0.3, 
                                                          random_state=None)
    KNN.fit(X_train, y_train)
    acc = KNN.score(X_valid, y_valid)
    accuracies.append(acc)
print(f'\nMean accuracy with KNN for validation set: {round(np.mean(accuracies), 2)}')