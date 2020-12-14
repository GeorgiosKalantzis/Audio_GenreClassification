import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,StratifiedKfold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from vecstack import stacking
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from 



#####################------------A NEURAL NETWORK CAN BE USED FOR LEVEL 1 OR 2 CLASSIFIER------------#####################
#############################################--------FIXED A BUG ON INFORMATION LEAKAGE------------#############################################

#Stacking Function.This function is used to make predictions on n_folds of train and test dataset.
#Returns the prediction for train and test set for each model


def Stacking(model,train,y,test,n_fold):

    folds=StratifiedKFold(n_splits=n_fold)
    train_pred=np.empty((0,1),int)
    test_pred=np.empty((0,1),int)

    for train_indices,val_indices in folds.split(train,y.values):
            x_train, x_val=train.iloc[train_indices],train.iloc[val_indices]
            y_train, y_val=y.iloc[train_indices],train.iloc[val_indices]
            model.fit(X=x_train,y=y_train.values)
            train_pred=np.append(train_pred,model.predict(x_val))
    test_pred=np.append(test_pred,model.predict(test))

    return test_pred.reshape(-1,1), train_pred







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

#For Easier Usage of the function we turn arrays into data frames
X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)
Y_train=pd.DataFrame(Y_train)
Y_test=pd.DataFrame(Y_test)




#CLASSIFIERS ARE TUNED
rfc=RandomForestClassifier(max_depth=150,max_features='sqrt',min_samples_leaf=1,min_samples_split=2,n_estimators=300)
svm=SVC(C=94,gamma=0.005,kernel='rbf',class_weight=None,decision_function_shape='ovr')
lr=LogisticRegression(max_iter=1000)
xgb=XGBClassifier(learning_rate=0.05,max_depth=3,n_estimators=1000,objective='multi:softmax')
knn=KNeighborsClassifier(n_neighbors=5)

#1st level classifiers
model1=svm
test_pred1,train_pred1=Stacking(model=model1,train=X_train,test=X_test,y=Y_train,n_fold=2)
train_pred1=pd.DataFrame(train_pred1)
test_pred1=pd.DataFrame(test_pred1)


model2=rfc
test_pred2,train_pred2=Stacking(model=model2,train=X_train,test=X_test,y=Y_train,n_fold=2)
train_pred2=pd.DataFrame(train_pred2)
test_pred2=pd.DataFrame(test_pred2)

model3=xgb
test_pred3,train_pred3=Stacking(model=model3,train=X_train,test=X_test,y=Y_train,n_fold=2)
train_pred3=pd.DataFrame(train_pred3)
test_pred3=pd.DataFrame(test_pred3)


df=pd.concat([train_pred1,train_pred2, train_pred3],axis=1)
df_test=pd.concat([test_pred1, test_pred3, test_pred3],axis=1)


#2nd level classifiers
model4=knn
test_pred4,train_pred4=Stacking(model=model4,train=df,test=df_test,y=Y_train,n_fold=2)
test_pred4=pd.DataFrame(test_pred4)
train_pred4=pd.DataFrame(train_pred4)

model5=LogisticRegression(max_iter=500)
test_pred5,train_pred5=Stacking(model=model5,train=df,test=df_test,y=Y_train,n_fold=2)
test_pred5=pd.DataFrame(test_pred5)
train_pred5=pd.DataFrame(train_pred5)
df=pd.concat([train_pred4,train_pred5],axis=1)
df_test=pd.concat([test_pred4,test_pred5],axis=1)
#Meta Classifier
model=RandomForestClassifier(max_depth=150,max_features='sqrt',min_samples_leaf=1,min_samples_split=2,n_estimators=300)
model.fit(df,Y_train)
print(model.score(df_test,Y_test))

