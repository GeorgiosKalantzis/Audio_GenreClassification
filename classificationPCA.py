# -*- coding: utf-8 -*-
"""
PCA

"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 



data = pd.read_csv('df_features.csv',header=None)

y = data.iloc[1:,0]
x = data.iloc[1:,1:]

# Standardizing the features

x = StandardScaler().fit_transform(x)

encoder = preprocessing.LabelEncoder()

y = encoder.fit_transform(y)

# Splittting them
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state = 2020, stratify=y)

pca = PCA(.95)

pca.fit(X_train)

X_train = pca.transform(X_train)
X_test = pca.transform(X_test)

# Building the SVM model 
svmclf = SVC() 
svmclf.fit(X_train, y_train) 
  
# Storing the predictions of the non-linear model 
y_pred_svmclf = svmclf.predict(X_test) 
  
# Evaluating the performance of the non-linear model 
print('Accuracy : '+str(accuracy_score(y_test, y_pred_svmclf)))









