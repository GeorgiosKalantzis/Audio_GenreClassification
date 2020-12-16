import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from glob import glob
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix,ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#reading the files and initializing the train and test sets
filename=glob('df_features*.csv')
dataframes=[pd.read_csv(f) for f in filename]
models=[0 for i in range(21)]
predictions=np.array([[0 for x in range(300)]for y in range(len(models))])
"""
pca=[0 for i in range(21)]
lda = [0 for i in range(21)]
"""


#creating 21 train and test sets for each model
for i in range(21):
    
    data_list = list(dataframes[i].columns)
    genre_list = dataframes[i].iloc[:,-1]
    encoder = LabelEncoder()
    y=encoder.fit_transform(genre_list)
    scaler=StandardScaler()
    x=scaler.fit_transform(np.array(dataframes[i].iloc[0:,:-1],dtype=float))
    X_train, X_test,Y_train,Y_test=train_test_split(x, y ,test_size=.30,random_state=0,stratify=y)
    models[i] = SVC(C=94, gamma=0.005, kernel='rbf', probability=True, decision_function_shape='ovr')
    """
    LDA
    
    lda[i] = LDA()
    X_train = lda[i].fit_transform(X_train, Y_train)
    X_test = lda[i].transform(X_test)
    
    PCA
    
    pca[i] = PCA(0.95)
    pca[i].fit(X_train)
    X_train = pca[i].transform(X_train)
    X_test = pca[i].transform(X_test)
    """
    #Fitting the models
    models[i].fit(X_train,Y_train)
    #Array to hold the predictions of models
    predictions[i,:]=models[i].predict(X_test)



# array with the final predictions
final_prediction=np.array([0 for x in range(len(Y_test))])



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

#Accuracy:With PCA:79%,Without PCA:82%
