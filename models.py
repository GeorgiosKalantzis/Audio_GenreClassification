"""
Classifiers
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
import numpy as np



    

train = pd.read_csv('df_features.csv',header=None)


y = train.iloc[1:,0]
x = train.iloc[1:,1:]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state = 2020, stratify=y)
   
y_train = np.array(y_train)
   
rfc = RandomForestClassifier()
   
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

print(round(rfc.score(X_test,y_test),4))
   

