import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from glob import glob
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.decomposition import PCA
#reading the files and initializing the train and test sets
filename=glob('df_features*.csv')
dataframes=[pd.read_csv(f) for f in filename]
X_train=[[0 for i in range(21)] for j in range(700)]
Y_train=[[0 for i in range(1)] for j in range(700)]
X_test=[[0 for i in range(21)] for j in range(300)]
Y_test=[[0 for i in range(1)] for j in range(300)]


#creating 21 train and test sets for each model
for i in range(21):
    data_list=list(dataframes[i].columns)
    genre_list=dataframes[i].iloc[:,-1]
    encoder=LabelEncoder()
    y=encoder.fit_transform(genre_list)
    scaler=StandardScaler()
    x=scaler.fit_transform(np.array(dataframes[i].iloc[0:,:-1],dtype=float))
    X_train[i], X_test[i],Y_train[i],Y_test[i]=train_test_split(x, y ,test_size=.30,random_state=0,stratify=y)

models=[0 for i in range(21)]

#creating the models


for i in range(21):
    models[i]=SVC(C=94,gamma=0.005,kernel='rbf',probability=True,decision_function_shape='ovr')
    #pca[i]=PCA(0.95)


final_prediction=[]
# array to hold the predictions of models

predictions=np.array([[0 for x in range(len(Y_test))]for y in range(len(models))])

# array with the final predictions
final_prediction=np.array([0 for x in range(len(Y_test))])

"""
----------------------------------
#PCA TO BE USED.THE ACCURACY WENT DOWN 3%.CHECK CORRELATION MATRIX TO SEE WHERE YOU GAIN AND WHERE YOU LOSE
pca=[0 for i in range(21)]
for i in range(21):
    pca[i]=PCA(0.95)
    pca[i].fit(X_train[i])
    X_train[i]=pca[i].transform(X_train[i])
    X_test[i]=pca[i].transform(X_test[i])
"""

# Training and Testing the models

for i in range(21):
    models[i].fit(X_train[i],Y_train[i])
    predictions[i,:]=models[i].predict(X_test[i])

for j in range(len(Y_test)):
        class_votes = np.array([0 for x in range(10)])

        for i in range(10):
            for w in range(21):

                #if the predicted class of model w and song j is the same as the class give 1 vote to this class

                if predictions[w,j]==i:
                    class_votes[i]=class_votes[i]+1

        #take the index of the most voted class
        index_max=np.argmax(class_votes)

        #final prediction of class
        final_prediction[j]=index_max


print(classification_report(Y_test[0],final_prediction))

#computing and plotting the confusion matrix

labels=['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
cm=confusion_matrix(Y_test[0],final_prediction,normalize='true')
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp=disp.plot()
plt.show()