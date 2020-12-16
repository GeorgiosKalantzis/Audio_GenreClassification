# -*- coding: utf-8 -*-
"""
Filter Methods
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('df_features.csv', header=0)
y = data.iloc[0:,0]
x = data.iloc[0:,1:]

# import and create the VarianceThreshold object.
from sklearn.feature_selection import VarianceThreshold
vs_constant = VarianceThreshold(threshold=0.001)

# fit the object to our data.
vs_constant.fit(x)

# get the constant column names.
constant_columns = [column for column in x.columns
                    if column not in x.columns[vs_constant.get_support()]]

# drop the constant columns
#x.drop(labels=constant_columns, axis=1, inplace=True)
#X_test.drop(labels=constant_columns, axis=1, inplace=True)

"""
With the second basic method, we remove the Quasi-Constant Features,
in which a value occupies the majority of the observations.
"""

# make a threshold for quasi constant.
threshold = 0.90 #a value occupies 90% of the observations for a specific feature.

# create empty list
quasi_constant_features = []

# loop over all the columns
for feature in x.columns:
    # calculate the ratio.
    predominant = (x[feature].value_counts() / np.float(len(x))).sort_values(ascending=False).values[0]
    # append the column name if it is bigger than the threshold
    if predominant >= threshold:
        quasi_constant_features.append(feature)

print(quasi_constant_features)

#drop the quasi constant columns
#X_train.drop(labels=quasi_constant_feature, axis=1, inplace=True)
#X_test.drop(labels=quasi_constant_feature, axis=1, inplace=True)
"""

With the third basic method, we remove the duplicate features.
"""
# transpose the feature matrice
train_features_T = x.T

# print the number of duplicated features
print(train_features_T.duplicated().sum())

# select the duplicated features columns names
duplicated_columns = train_features_T[train_features_T.duplicated()].index.values

# drop those columns
#X_train.drop(labels=duplicated_columns, axis=1, inplace=True)
#X_test.drop(labels=duplicated_columns, axis=1, inplace=True)


"""
Correlated filter methods.
Features should be highly corellated with the target.
Features should be uncorellated among themselves.
"""

import seaborn as sns
import matplotlib.pyplot as plt

#Create a set to hold the correlated features
corr_features = set()

#Create the correlation matrix
corr_matrix = x.corr(method='kendall') #pearson, kendall or spearman

#display a heatmap of the correlation matrix
plt.figure(figsize=(18,8))
sns.heatmap(corr_matrix)


for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if(corr_matrix.iloc[i,j]) > 0.9 and (i != j):
            colname = corr_matrix.columns[i]
            corr_features.add(colname)



"""

Statistical & Ranking Filter Methods.
These methods rank the features based on certain criteria or metrics,
then select the features with the highest ranking.
"""
"""
Information gain is used to measure the dependence between features and labels and
calculates the information gain between the i-th feature and the class labels.
In information gain, a feature is relevant if it has a high information gain.
"""

from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

importances = mutual_info_classif(x,y)
feat_importances = pd.Series(importances, x.columns[0:len(x.columns)])
feat_importances.plot(kind='barh', color='blue') #plot info gain of each features
plt.show()

# select the number of features you want to retain.
select_k = 10 #whatever we want

# create the SelectKBest with the mutual info(info gain) strategy.
selection = SelectKBest(mutual_info_classif, k=select_k).fit(x, y)

#plot the scores
plt.bar([i for i in range(len(selection.scores_))], selection.scores_)
plt.show()

# display the retained features.
features = x.columns[selection.get_support()]
print(features)







