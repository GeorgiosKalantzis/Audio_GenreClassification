
"""
Autoencoder
"""

import pandas as pd  
import numpy as np 
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import MinMaxScaler  
from keras.layers import Input, Dense 
from keras.models import Model, Sequential 
from keras import regularizers
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from keras import layers


data = pd.read_csv('df_features.csv', header = None)

y = data.iloc[1:,0]


encoder = preprocessing.LabelEncoder()

y = encoder.fit_transform(y)


X = MinMaxScaler().fit_transform(np.array(data.iloc[1:,1:])) 



# Building the Input Layer 
input_layer = Input(shape =(X.shape[1], )) 
  
# Building the Encoder network
 
encoded = Dense(120, activation ='tanh', 
                activity_regularizer = regularizers.l1(10e-5))(input_layer) 
encoded = Dense(100, activation ='tanh', 
                activity_regularizer = regularizers.l1(10e-5))(encoded)


decoded = Dense(120, activation ='tanh')(encoded)

  
# Building the Output Layer 
output_layer = Dense(X.shape[1], activation ='relu')(decoded)


# Defining the parameters of the Auto-encoder network 
autoencoder = Model(input_layer, output_layer) 
autoencoder.compile(optimizer ="adadelta", loss ="mse") 
  
# Training the Auto-encoder network 
autoencoder.fit(X, X,  
                batch_size = 16, epochs = 10,  
                shuffle = True, validation_split = 0.20)



hidden_representation = Sequential() 
hidden_representation.add(autoencoder.layers[0]) 
hidden_representation.add(autoencoder.layers[1]) 
hidden_representation.add(autoencoder.layers[2]) 
 

"""

----------- FINE TUNING -------------------------------------

hidden_representation.add(layers.Dense(60, activation='relu'))
hidden_representation.add(layers.Dense(30, activation='relu'))
hidden_representation.add(layers.Dense(20, activation='relu'))
hidden_representation.add(layers.Dense(10, activation='softmax'))
hidden_representation.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

classifier = hidden_representation.fit(X_train,y_train,epochs = 30 , batch_size = 128)

# Accuracy
test_loss, test_acc  = hidden_representation.evaluate(X_test, y_test, batch_size=128)

print(round(test_acc,4))
print(round(test_loss,4))




"""

# Generate new dataset with less features
encoded_X = hidden_representation.predict(X)

# Splitting the encoded data for linear classification 
X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(encoded_X, y, test_size = 0.2)

# Building the SVM model 
svmclf = SVC() 
svmclf.fit(X_train_encoded, y_train_encoded) 
  
# Storing the predictions of the non-linear model 
y_pred_svmclf = svmclf.predict(X_test_encoded) 
  
# Evaluating the performance of the non-linear model 
print('Accuracy : '+str(accuracy_score(y_test_encoded, y_pred_svmclf)))

# Neural network
model = Sequential()
model.add(layers.Dense(60, activation='relu', input_shape=(X_train_encoded.shape[1],)))
model.add(layers.Dense(30, activation='relu'))

model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


classifier = model.fit(X_train_encoded,
                    y_train_encoded,
                    epochs=450,
                    batch_size=128)

# Accuracy
test_loss, test_acc  = model.evaluate(X_test_encoded, y_test_encoded, batch_size=128)

print(round(test_acc,4))





