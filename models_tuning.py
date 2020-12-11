import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from vecstack import stacking
from xgboost import XGBClassifier
import scipy
from sklearn.metrics import accuracy_score, classification_report


train = pd.read_csv('df_features.csv',header=0)
y=train.iloc[0:,0]
x=train.iloc[0:,1:]

encoder = LabelEncoder()
y = encoder.fit_transform(y)
scaler = StandardScaler()

x = scaler.fit_transform(np.array(train.iloc[:, :-1], dtype=float))
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=2020, stratify=y)

#TUNING XGB
n_estimators=[int(x) for x in np.linspace(start=10,stop=1000, num=10)]
max_depth=[int(x) for x in np.linspace(3,110,num=11)]


param_grid = {'n_estimators': n_estimators, 'learning_rate': [0.08, 0.01,0.1], 'max_depth':max_depth,'objective':['multi:softmax']}
model = RandomizedSearchCV(
    estimator=XGBClassifier(), param_distributions=param_grid, cv=5, verbose=20)
y_xgb_grid=model.fit(X_train,Y_train)
print(model.best_params_)


#TUNING RANDOM FOREST
n_estimators=[int(x) for x in np.linspace(start=200,stop=2000, num=10)]
max_features=['auto','sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap=[True, False]

random_grid={'n_estimators':n_estimators, 'max_features':max_features,
             'max_depth':max_depth,'min_samples_split':min_samples_split,
             'min_samples_leaf':min_samples_leaf, 'bootstrap':bootstrap}
grid2=RandomizedSearchCV(RandomForestClassifier(),param_distributions=random_grid,n_iter=100, cv=3, verbose=2,
                         random_state=42, n_jobs=-1)
grid2.fit(X_train,Y_train)
print(grid2.best_params_)


#TUNING SVC
param_grid={'C':scipy.stats.expon(scale=100),'gamma':scipy.stats.expon(scale=.1),
            'kernel':['rbf','linear','poly','sigmoid'],'class_weight':['balanced',None],'degree':[1, 2, 5, 8, 10]}
model=RandomizedSearchCV(SVC(),param_distributions=param_grid,n_iter=100,cv=3,verbose=2,random_state=42,n_jobs=-1)
model.fit(X_train,Y_train)
print(model.best_params_)





