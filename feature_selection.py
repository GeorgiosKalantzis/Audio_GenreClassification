"""
Basic filter methods.
With the first basic method, we remove constant features.
Constant Features show single values in all the observations in the dataset,
providing no useful info.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
train = pd.read_csv('df_features.csv', header=0)
y = train.iloc[0:,0]
x = train.iloc[0:,1:]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=2020, stratify=y)

# import and create the VarianceThreshold object.
from sklearn.feature_selection import VarianceThreshold
vs_constant = VarianceThreshold(threshold=0)

# fit the object to our data.
vs_constant.fit(X_train)

# get the constant column names.
constant_columns = [column for column in X_train.columns
                    if column not in X_train.columns[vs_constant.get_support()]]

# drop the constant columns
#X_train.drop(labels=constant_columns, axis=1, inplace=True)
#X_test.drop(labels=constant_columns, axis=1, inplace=True)

"""
With the second basic method, we remove the Quasi-Constant Features,
in which a value occupies the majority of the observations.
"""
import numpy as np
# make a threshold for quasi constant.
threshold = 0.90 #a value occupies 90% of the observations for a specific feature.

# create empty list
quasi_constant_features = []

# loop over all the columns
for feature in X_train.columns:
    # calculate the ratio.
    predominant = (X_train[feature].value_counts() / np.float(len(X_train))).sort_values(ascending=False).values[0]
    # append the column name if it is bigger than the threshold
    if predominant >= threshold:
        quasi_constant_feature.append(feature)

print(quasi_constant_feature)

#drop the quasi constant columns
#X_train.drop(labels=quasi_constant_feature, axis=1, inplace=True)
#X_test.drop(labels=quasi_constant_feature, axis=1, inplace=True)

"""
With the third basic method, we remove the duplicate features.
"""
# transpose the feature matrice
train_features_T = X_train.T

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
corr_matrix = X_train.corr(method='spearman') #pearson, kendall or spearman

#display a heatmap of the correlation matrix
plt.figure(figsize=(18,8))
sns.heatmap(corr_matrix)


for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if(corr_matrix.iloc[i,j]) > 0.9 and (i != j):
            colname = corr_matrix.columns[i]
            corr_features.add(colname)

#drop highly corellated features
#X_train.drop(labels=corr_features, axis=1, inplace=True)
#X_test.drop(labels=corr_features, axis=1, inplace=True)


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

importances = mutual_info_classif(X_train,y_train)
feat_importances = pd.Series(importances, X_train.columns[0:len(X_train.columns)])
feat_importances.plot(kind='barh', color='blue') #plot info gain of each features
plt.show()

# select the number of features you want to retain.
select_k = 30 #whatever we want

# create the SelectKBest with the mutual info(info gain) strategy.
selection = SelectKBest(mutual_info_classif, k=select_k).fit(X_train, y_train)

#plot the scores
plt.bar([i for i in range(len(selection.scores_))], selection.scores_)
plt.show()

# display the retained features.
features = X_train.columns[selection.get_support()]
print(features)

###*** We have to drop the other features(with less info gain from X_train,X_test) ***###

"""
Feature importance based on ROC-AUC metrics.
AUC–ROC curve is used in bi–multi class classification problems.
"""

# import the DecisionTree Algorithm and evaluation score.
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# list of the resulting scores.
roc_values = []

# loop over all features and calculate the score.
for feature in x_train.columns:
    clf = DecisionTreeClassifier()
    clf.fit(x_train[feature].to_frame(), y_train)
    y_scored = clf.predict_proba(x_test[feature].to_frame())
    roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))

# create a Pandas Series for visualisation.
roc_values = pd.Series(roc_values)
roc_values.index = X_train.columns

# show the results.
print(roc_values.sort_values(ascending=False))

###*** We have to drop the other features(with less info gain from X_train,X_test) ***###

"""
WRAPPER METHODS.
Wrapper methods work by evaluating a subset of features using a machine learning algorithm
that employs a search strategy to look through the space of possible feature subsets,
evaluating each subset based on the quality of the performance of a given algorithm.
"""
"""
FORWARD FEATURE SELECTION.
an iterative method in which we start by evaluating all features individually,
and then select the one that results in the best performance.
"""

from mlxtend.feature_selection import SequentialFeatureSelector

# import the algorithm you want to evaluate on your features.
from sklearn.ensemble import RandomForestClassifier

# create the SequentialFeatureSelector object, and configure the parameters.
sfs = SequentialFeatureSelector(RandomForestClassifier(),
           k_features=10,
           forward=True,
           scoring='accuracy',
           cv=2)

# fit the object to the training data.
sfs = sfs.fit(X_train, y_train)

# print the selected features.
selected_features = X_train.columns[list(sfs.k_feature_idx_)]
print('selected features from forward selection:', selected_features)

# print the final prediction score.
print('accuracy:', sfs.k_score_)

# transform to the newly selected features.
#X_train = sfs.transform(X_train)
#X_test = sfs.transform(X_test)

"""
BACKWARD FEATURE ELIMINATION.
At each iteration, backward feature elimination removes one feature at a time.
This feature can be also described as the least significant feature among
the remaining available ones.
"""

# just set forward=False for backward feature selection.
# create theSequentialFeatureSelector object, and configure the parameters.
sbs = SequentialFeatureSelector(RandomForestClassifier(),
           k_features=10,
           forward=False,
           scoring='accuracy',
           cv=2)

# fit the object to our training data.
sbs = sbs.fit(X_train, y_train)

# print the selected features.
selected_features = X_train.columns[list(sbs.k_feature_idx_)]
print('selected features from backward selection:', selected_features)

# print the final prediction score.
print('accuracy:', sbs.k_score_)

# transform to the newly selected features.
#X_train = sbs.transform(X_train)
#X_test = sbs.transform(X_test)

"""
EXHAUSTIVE FEATURE SELECTION.
This method searches across all possible feature combinations.
Its aim is to find the best performing feature subset.
"""
# import the algorithm you want to evaluate on your features.
from mlxtend.feature_selection import ExhaustiveFeatureSelector

# create the ExhaustiveFeatureSelector object.
efs = ExhaustiveFeatureSelector(RandomForestClassifier(),
           min_features=4,
           max_features=10,
           scoring='accuracy',
           cv=2)

# fit the object to the training data.
efs = efs.fit(X_train, y_train)

# print the selected features.
selected_features = X_train.columns[list(efs.k_feature_idx_)]
print('selected features from exhaustive selection:', selected_features)

# print the final prediction score.
print('accuracy:', efs.k_score_)

# transform to the newly selected features.
#X_train = efs.transform(X_train)
#X_test = efs.transform(X_test)
