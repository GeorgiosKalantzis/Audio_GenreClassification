import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from vecstack import stacking
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier



#####################------------A NEURAL NETWORK CAN BE USED FOR LEVEL 1 OR 2 CLASSIFIER------------#####################
train = pd.read_csv('df_features.csv')
y = train.iloc[1:,0]
x = train.iloc[1:,1:]

data_list = list(train.columns)

#Encoding the labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)
#Normalizing Data
scaler = StandardScaler()
x = scaler.fit_transform(np.array(train.iloc[1:, 1:], dtype=float))
#Splitting the dataset in training and testing
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.30, random_state=2020, stratify=y)


#CLASSIFIERS ARE TUNED
rfc=RandomForestClassifier(max_depth=150,max_features='sqrt',min_samples_leaf=1,min_samples_split=2,n_estimators=300)
svm=SVC(C=94,gamma=0.005,kernel='rbf',class_weight=None,decision_function_shape='ovr')
lr=LogisticRegression(max_iter=1000)
xgb=XGBClassifier(learning_rate=0.05,max_depth=3,n_estimators=1000,objective='multi:softmax')
knn=KNeighborsClassifier(n_neighbors=5)

#Fitting and Predicting with 1st level classifiers
rfc.fit(X_train,Y_train)
svm.fit(X_train,Y_train)
lr.fit(X_train,Y_train)

f1=rfc.predict(X_test)
f2=svm.predict(X_test)
f3=lr.predict(X_test)


#Combining Predictions into an array
f=[f1, f2, f3]
f=np.transpose(f)
#####------------------#####

#Fitting and Predicting with 2nd level classifiers with the predicted Values from above
knn.fit(f,Y_test)
svm.fit(f,Y_test)

final1=knn.predict(f)
final2=svm.predict(f)


#Combining Predictions into an array
final=[final1, final2]
final=np.transpose(final)
#####------------#####



#Fitting and Predicting with 3rd level classifier
xgb.fit(final, Y_test)
pred=xgb.predict(final)


#Printing the results from each stage
print(classification_report(Y_test,f1))
print(classification_report(Y_test,f2))
print(classification_report(Y_test,f3))
print(classification_report(Y_test,final1))
print(classification_report(Y_test,final2))
print(classification_report(Y_test,pred))

###########10% better result(80% accuracy) at last instead of predicting with one classifier#########
