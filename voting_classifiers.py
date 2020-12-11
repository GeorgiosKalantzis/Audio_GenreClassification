import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,NuSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,plot_confusion_matrix

#reading the dataset
train = pd.read_csv('df_features.csv')

y = train.iloc[1:,0]
x = train.iloc[1:,1:]

data_list = list(train.columns)


encoder = LabelEncoder()
y = encoder.fit_transform(genre_list)
scaler = StandardScaler()
x = scaler.fit_transform(np.array(train.iloc[:, :-1], dtype=float))
#splitting to training and test
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=2020, stratify=y)


classifier1=SVC(C=94.04920709825932, gamma=0.005285108738039248,kernel="rbf",probability=True,decision_function_shape="ovr"
                ,class_weight=None)
classifier2=RandomForestClassifier(bootstrap=False, max_depth=150, max_features='sqrt',
min_samples_leaf=1, min_samples_split=2, n_estimators= 300)
classifier3=LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000,multi_class='multinomial')
classifier4=XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1, n_estimators=100, max_depth=3)
classifier5=NuSVC(gamma=0.005,kernel="rbf",nu=0.5,
                  class_weight=None,probability=True,decision_function_shape='ovr')

estimators=[]
estimators.append(('SVC', classifier1))
estimators.append(('RF', classifier2))
estimators.append(('LR', classifier3))
estimators.append(('XGB', classifier4))
estimators.append(('NuSVC',classifier5))



vot_hard=VotingClassifier(estimators=estimators, voting='hard',verbose=False)
vot_hard.fit(X_train,Y_train)
y_pred=vot_hard.predict(X_test)
score=accuracy_score(Y_test,y_pred)
print("Hard voting score " ,score)
vot_soft=VotingClassifier(estimators=estimators,voting='soft',verbose=False)
vot_soft.fit(X_train,Y_train)
y_pred=vot_soft.predict(X_test)
score=accuracy_score(Y_test,y_pred)
print("Soft voting score ",score)
cm=confusion_matrix(Y_test,y_pred)
accuracy=vot_soft.score(X_test,Y_test)
disp=plot_confusion_matrix(vot_soft, X_test,Y_test,normalize='true')
plt.show()
