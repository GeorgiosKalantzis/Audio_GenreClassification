"""
Classifiers
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras import layers
import keras
from sklearn import svm

# Dictionary to hold the results
models_accuracy = {'Random Forest': 0 , 'ANN' : 0 , 'SVM' : 0}

    
# Read data
data = pd.read_csv('df_features.csv',header=None)


y = data.iloc[1:,0]
x = data.iloc[1:,1:]

# -------------------------------- Random Forest ------------------------------------------------
# Split them
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 2020, stratify=y)
   
y_train = np.array(y_train)

# Prediction using RandomForest   
rfc = RandomForestClassifier()
   
rfc.fit(X_train,y_train)

y_pred = rfc.predict(X_test)

# Accuracy
models_accuracy['Random Forest'] = round(rfc.score(X_test,y_test),4)
# ------------------------------------------------------------------------------------------------

# ----------------------------- Neural Network ---------------------------------------------------
# Encoding the targets for the NN
encoder = preprocessing.LabelEncoder()

y = encoder.fit_transform(y)

# Scaling them
scaler = StandardScaler()

X = scaler.fit_transform(np.array(data.iloc[1:,1:]))

# Splittting them
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Neural network
model = Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


classifier = model.fit(X_train,
                    y_train,
                    epochs=250,
                    batch_size=128)

# Accuracy
test_loss, test_acc  = model.evaluate(X_test, y_test, batch_size=128)
models_accuracy['ANN'] = round(test_acc,4)

# -------------------------------------------------------------------------------------------------

# ------------------------------------- SVM ------------------------------------------------------

# Linear SVM
clf = svm.SVC(kernel = 'linear')

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)



models_accuracy['SVM'] = round(metrics.accuracy_score(y_test, y_pred),4)

# ---------------------------------------------------------------------------------------------------
# Final results
print(models_accuracy)


# Results -> Random Forest : 69%, ANN : 78%, SVM: 75% 
   

