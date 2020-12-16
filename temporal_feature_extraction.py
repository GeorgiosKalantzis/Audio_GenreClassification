import librosa, librosa.display, librosa.feature
import pandas as pd
import numpy as np
import scipy
import pandas as pd
from scipy.stats import gmean,moment
import os
import csv



################-----------------TEMPORAL FEATURES THAT ARE BEING EXTRACTED-----------------##################
################ MEAN AND STD OF:ZCR,RMS,SPECTRAL CENTROID,SPECTRAL ROLLOFF,SPECTRAL BANDWIDTH2,MFCC,SPECTRAL CONTRAST ################
################ RSD,HCF,LCF OF:ZCR,RMS,SPECTRAL CENTROID,SPECTRAL ROLLOFF,SPECTRAL BANDWIDTH2 ##############
########################-----------------TEMPORAL FLATNESS FOR EVERY WINDOW-----------------################
# Init dataframe to hold the features
df1 = pd.DataFrame()
df2 = pd.DataFrame()
df3 = pd.DataFrame()
df4 = pd.DataFrame()
df5 = pd.DataFrame()
df6 = pd.DataFrame()
df7 = pd.DataFrame()
df8 = pd.DataFrame()
df9 = pd.DataFrame()
df10 = pd.DataFrame()
df11 = pd.DataFrame()
df12 = pd.DataFrame()
df13 = pd.DataFrame()
df14 = pd.DataFrame()
df15 = pd.DataFrame()
df16 = pd.DataFrame()
df17 = pd.DataFrame()
df18 = pd.DataFrame()
df19 = pd.DataFrame()
df20 = pd.DataFrame()
df21 = pd.DataFrame()
"""
df22 = pd.DataFrame()
df23 = pd.DataFrame()
df24 = pd.DataFrame()
"""

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

for g in genres:
     for filename in os.listdir(f'./genres/{g}'):

        songname = f'./genres/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, sr=22050)
        # feature list of 20 windows*25 statistics.rows are the windows columns are the statistics
        print(filename)
        feature_list= [[0 for i in range(71)] for j in range(21)]
        counter=0
        counter2=0
        spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr, n_bands=6, fmin=200.0)
        zcr = librosa.feature.zero_crossing_rate(y + 0.0001, frame_length=2048, hop_length=512)[0]
        rms = librosa.feature.rms(y + 0.0001)[0]
        spectral_centroids = librosa.feature.spectral_centroid(y + 0.01, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y + 0.01, sr=sr, roll_percent=0.85)[0]
        spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y + 0.01, sr=sr, p=2)[0]
        mfccs=librosa.feature.mfcc(y, sr=sr, n_mfcc=13)
        for i in range(21):
            #the last window is going to be longer
            if i==20:
                counter2=(i+1)*61+13

            else:
                counter2=(i+1)*61

            #zcr statistics
            feature_list[i][0]=np.std(zcr[counter:counter2])
            feature_list[i][1]=np.mean(zcr[counter:counter2])
            feature_list[i][2]=(np.mean(zcr[counter:counter2])/np.std(zcr[counter:counter2]))
            feature_list[i][3]=(np.max(zcr[counter:counter2])/(np.mean(zcr[counter:counter2])))
            feature_list[i][4]=(np.min(zcr[counter:counter2])/(np.mean(zcr[counter:counter2])))

            #rms statistics
            feature_list[i][5]=np.std(rms[counter:counter2])
            feature_list[i][6]=np.mean(rms[counter:counter2])
            feature_list[i][7]=np.mean(rms[counter:counter2])/np.std(rms[counter:counter2])
            feature_list[i][8]=np.max(rms[counter:counter2]) / np.mean(rms[counter:counter2])
            feature_list[i][9]=np.min(rms[counter:counter2]) / np.mean(rms[counter:counter2])

            #centroids statistics
            feature_list[i][10]=np.std(spectral_centroids[counter:counter2])
            feature_list[i][11]=np.mean(spectral_centroids[counter:counter2])
            feature_list[i][12]=(np.mean(spectral_centroids[counter:counter2]) / np.std(spectral_centroids[counter:counter2]))
            feature_list[i][13]=(np.max(spectral_centroids[counter:counter2]) / np.mean(spectral_centroids[counter:counter2]))
            feature_list[i][14]=(np.min(spectral_centroids[counter:counter2]) / np.mean(spectral_centroids[counter:counter2]))

            #spectral rolloff statistics
            feature_list[i][15]=np.std(spectral_rolloff[counter:counter2])
            feature_list[i][16]=np.mean(spectral_rolloff[counter:counter2])
            feature_list[i][17]=(np.mean(spectral_rolloff[counter:counter2])/np.std(spectral_rolloff[counter:counter2]))
            feature_list[i][18]=np.max(spectral_rolloff[counter:counter2])/np.mean(spectral_rolloff[counter:counter2])
            feature_list[i][19]=np.min(spectral_rolloff[counter:counter2]) / np.mean(spectral_rolloff[counter:counter2])

            #spectral bandwidth statistics
            feature_list[i][20] = np.std(spectral_bandwidth_2[counter:counter2])
            feature_list[i][21] = np.mean(spectral_bandwidth_2[counter:counter2])
            feature_list[i][22] = (np.mean(spectral_bandwidth_2[counter:counter2]) / np.std(spectral_bandwidth_2[counter:counter2]))
            feature_list[i][23] = (np.max(spectral_bandwidth_2[counter:counter2]) / np.mean(spectral_bandwidth_2[counter:counter2]))
            feature_list[i][24] = (np.min(spectral_bandwidth_2[counter:counter2]) / np.mean(spectral_bandwidth_2[counter:counter2]))
            #mfccs statistics
            #1st
            feature_list[i][25] = np.mean(mfccs[0,counter:counter2])
            feature_list[i][26] = np.std(mfccs[0,counter:counter2])
            #2nd
            feature_list[i][27] = np.mean(mfccs[1,counter:counter2])
            feature_list[i][28] = np.std(mfccs[1, counter:counter2])
            #3d
            feature_list[i][29] = np.mean(mfccs[2, counter:counter2])
            feature_list[i][30] = np.std(mfccs[2, counter:counter2])
            #4th
            feature_list[i][31] = np.mean(mfccs[3, counter:counter2])
            feature_list[i][32] = np.std(mfccs[3, counter:counter2])
            #5th
            feature_list[i][33] = np.mean(mfccs[4, counter:counter2])
            feature_list[i][34] = np.std(mfccs[4, counter:counter2])
            #6th
            feature_list[i][35] = np.mean(mfccs[5, counter:counter2])
            feature_list[i][36] = np.std(mfccs[5, counter:counter2])
            #7th
            feature_list[i][37] = np.mean(mfccs[6, counter:counter2])
            feature_list[i][38] = np.std(mfccs[6, counter:counter2])
            #8th
            feature_list[i][39] = np.mean(mfccs[7, counter:counter2])
            feature_list[i][40] = np.std(mfccs[7, counter:counter2])
            #9th
            feature_list[i][41] = np.mean(mfccs[8, counter:counter2])
            feature_list[i][42] = np.std(mfccs[8, counter:counter2])
            #10th
            feature_list[i][43] = np.mean(mfccs[9, counter:counter2])
            feature_list[i][44] = np.std(mfccs[9, counter:counter2])
            #11th
            feature_list[i][45] = np.mean(mfccs[10, counter:counter2])
            feature_list[i][46] = np.std(mfccs[10, counter:counter2])
            #12th
            feature_list[i][47] = np.mean(mfccs[11, counter:counter2])
            feature_list[i][48] = np.std(mfccs[11, counter:counter2])
            #13th
            feature_list[i][49] = np.mean(mfccs[12, counter:counter2])
            feature_list[i][50] = np.std(mfccs[12, counter:counter2])

            #Spectral Contrast Statistics
            #1st
            feature_list[i][51] = np.mean(spectral_contrast[0, counter:counter2])
            feature_list[i][52] = np.std(spectral_contrast[0, counter:counter2])
            # 2nd
            feature_list[i][53] = np.mean(spectral_contrast[1, counter:counter2])
            feature_list[i][54] = np.std(spectral_contrast[1, counter:counter2])
            # 3d
            feature_list[i][55] = np.mean(spectral_contrast[2, counter:counter2])
            feature_list[i][56] = np.std(spectral_contrast[2, counter:counter2])
            # 4th
            feature_list[i][57] = np.mean(spectral_contrast[3, counter:counter2])
            feature_list[i][58] = np.std(spectral_contrast[3, counter:counter2])
            # 5th
            feature_list[i][59] = np.mean(spectral_contrast[4, counter:counter2])
            feature_list[i][60] = np.std(spectral_contrast[4, counter:counter2])
            # 6th
            feature_list[i][61] = np.mean(spectral_contrast[5, counter:counter2])
            feature_list[i][62] = np.std(spectral_contrast[5, counter:counter2])
            # 7th
            feature_list[i][63] = np.mean(spectral_contrast[6, counter:counter2])
            feature_list[i][64] = np.std(spectral_contrast[6, counter:counter2])

            #Temporal Flatness

            feature_list[i][65] = (gmean(zcr[counter:counter2])/np.mean(zcr[counter:counter2]))
            feature_list[i][66] = (gmean(rms[counter:counter2])/np.mean(rms[counter:counter2]))
            feature_list[i][67] = (gmean(spectral_centroids[counter:counter2])/np.mean(spectral_centroids[counter:counter2]))
            feature_list[i][68] = (gmean(spectral_rolloff[counter:counter2])/np.mean(spectral_rolloff[counter:counter2]))
            feature_list[i][69] = (gmean(spectral_bandwidth_2[counter:counter2])/np.mean(spectral_bandwidth_2[counter:counter2]))


            counter = 61 * (i + 1)
            feature_list[i][0:70] = np.round(feature_list[i][0:70], decimals=3)
            #add the class of the song
            feature_list[i][70] =g

        #pass the values of each window to the responding dataframe and create csv file for each window
        df1 = df1.append(pd.DataFrame(feature_list[0][:]).transpose(), ignore_index=True)
        df2 = df2.append(pd.DataFrame(feature_list[1][:]).transpose(), ignore_index=True)
        df3 = df3.append(pd.DataFrame(feature_list[2][:]).transpose(), ignore_index=True)
        df4 = df4.append(pd.DataFrame(feature_list[3][:]).transpose(), ignore_index=True)
        df5 = df5.append(pd.DataFrame(feature_list[4][:]).transpose(), ignore_index=True)
        df6 = df6.append(pd.DataFrame(feature_list[5][:]).transpose(), ignore_index=True)
        df7 = df7.append(pd.DataFrame(feature_list[6][:]).transpose(), ignore_index=True)
        df8 = df8.append(pd.DataFrame(feature_list[7][:]).transpose(), ignore_index=True)
        df9 = df9.append(pd.DataFrame(feature_list[8][:]).transpose(), ignore_index=True)
        df10 = df10.append(pd.DataFrame(feature_list[9][:]).transpose(), ignore_index=True)
        df11 = df11.append(pd.DataFrame(feature_list[10][:]).transpose(), ignore_index=True)
        df12 = df12.append(pd.DataFrame(feature_list[11][:]).transpose(), ignore_index=True)
        df13 = df13.append(pd.DataFrame(feature_list[12][:]).transpose(), ignore_index=True)
        df14 = df14.append(pd.DataFrame(feature_list[13][:]).transpose(), ignore_index=True)
        df15 = df15.append(pd.DataFrame(feature_list[14][:]).transpose(), ignore_index=True)
        df16 = df16.append(pd.DataFrame(feature_list[15][:]).transpose(), ignore_index=True)
        df17 = df17.append(pd.DataFrame(feature_list[16][:]).transpose(), ignore_index=True)
        df18 = df18.append(pd.DataFrame(feature_list[17][:]).transpose(), ignore_index=True)
        df19 = df19.append(pd.DataFrame(feature_list[18][:]).transpose(), ignore_index=True)
        df20 = df20.append(pd.DataFrame(feature_list[19][:]).transpose(), ignore_index=True)
        df21 = df21.append(pd.DataFrame(feature_list[20][:]).transpose(), ignore_index=True)
        """
        df22 = df22.append(pd.DataFrame(feature_list[21][:]).transpose(), ignore_index=True)
        df23 = df23.append(pd.DataFrame(feature_list[22][:]).transpose(), ignore_index=True)
        df24 = df24.append(pd.DataFrame(feature_list[23][:]).transpose(), ignore_index=True)
        """

        df1.to_csv('df_features1.csv', index=False)
        df2.to_csv('df_features2.csv',index=False)
        df3.to_csv('df_features3.csv',index=False)
        df4.to_csv('df_features4.csv',index=False)
        df5.to_csv('df_features5.csv',index=False)
        df6.to_csv('df_features6.csv',index=False)
        df7.to_csv('df_features7.csv',index=False)
        df8.to_csv('df_features8.csv',index=False)
        df9.to_csv('df_features9.csv',index=False)
        df10.to_csv('df_features10.csv',index=False)
        df11.to_csv('df_features11.csv',index=False)
        df12.to_csv('df_features12.csv',index=False)
        df13.to_csv('df_features13.csv',index=False)
        df14.to_csv('df_features14.csv',index=False)
        df15.to_csv('df_features15.csv',index=False)
        df16.to_csv('df_features16.csv',index=False)
        df17.to_csv('df_features17.csv',index=False)
        df18.to_csv('df_features18.csv',index=False)
        df19.to_csv('df_features19.csv',index=False)
        df20.to_csv('df_features20.csv',index=False)
        df21.to_csv('df_features21.csv',index=False)
        """
        df22.to_csv('df_features22.csv',index=False)
        df23.to_csv('df_features23.csv',index=False)
        df24.to_csv('df_features24.csv',index=False)
        """