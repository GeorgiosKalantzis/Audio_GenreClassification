import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
data = pd.read_csv('df_features.csv', header=None)
y = data.iloc[1:,0]
x = data.iloc[1:,1:]


"""

"""
"""
WRAPPER METHODS.
Wrapper methods work by evaluating a subset of features using a machine learning algorithm
that employs a search strategy to look through the space of possible feature subsets,
evaluating each subset based on the quality of the performance of a given algorithm.
"""


"""
EXHAUSTIVE FEATURE SELECTION.
This method searches across all possible feature combinations.
Its aim is to find the best performing feature subset.
"""
# import the algorithm you want to evaluate on your features.
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestClassifier

# create the ExhaustiveFeatureSelector object.
efs = ExhaustiveFeatureSelector(RandomForestClassifier(),
           min_features=45,
           max_features=70,
           scoring='accuracy',
           cv=2)

# fit the object to the training data.
efs = efs.fit(x, y)

# print the selected features.
selected_features1 = x.columns[list(efs.k_feature_idx_)]
print('selected features from exhaustive selection:', selected_features1)

# print the final prediction score.
print('accuracy:', efs.k_score_)

# transform to the newly selected features.
#X_train = efs.transform(X_train)
#X_test = efs.transform(X_test)

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
           k_features=60,
           forward=True,
           scoring='accuracy',
           cv=2)

# fit the object to the training data.
sfs = sfs.fit(x, y)

# print the selected features.
selected_features2 = x.columns[list(sfs.k_feature_idx_)]
print('selected features from forward selection:', selected_features2)

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
           k_features=60,
           forward=False,
           scoring='accuracy',
           cv=100)

# fit the object to our training data.
sbs = sbs.fit(x, y)

# print the selected features.
selected_features3 = x.columns[list(sbs.k_feature_idx_)]
print('selected features from backward selection:', selected_features3)

# print the final prediction score.
print('accuracy:', sbs.k_score_)

# transform to the newly selected features.
#X_train = sbs.transform(X_train)
#X_test = sbs.transform(X_test)
