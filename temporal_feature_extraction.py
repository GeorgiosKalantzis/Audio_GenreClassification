# -*- coding: utf-8 -*-
"""
Temporal Feature Extraction

"""

import librosa, librosa.display,librosa.feature
import pandas as pd
import numpy as np
import scipy

from tqdm import tqdm
import os

WAV_DIR = 'genres/'

# Column names of all the features that will be extracted
col_names = ['class','temp_zcr_std','temp_zcr_mean','temp_zcr_rsd','temp_zcr_hcf','temp_zcr_lcf','temp_rms_std','temp_rms_mean',
             'temp_rms_rsd','temp_rms_hcf','temp_rms_lcf','temp_spectral_centroid_mean','temp_spectral_centroid_std',
             'temp_spectral_centroid_rsd','temp_spectral_centroid_hcf','temp_spectral_centroid_lcf','temp_spectral_rolloff_mean', 'temp_spectral_rolloff_std',
            'temp_spectral_rolloff_rsd','temp_spectral_rolloff_hcf','temp_spectral_rolloff_lcf',
            'temp_spectral_bandwidth_2_mean', 'temp_spectral_bandwidth_2_std','temp_spectral_bandwidth_2_rsd',
            'temp_spectral_bandwidth_2_hcf','temp_spectral_bandwidth_2_lcf'
            ]

# Init dataframe to hold the features
df1 = pd.DataFrame(columns=col_names) # 1st window
df2 = pd.DataFrame(columns=col_names) # 2nd window
df3 = pd.DataFrame(columns=col_names) # 3rd window




# Parsing throught the file system to extract features from audio
for root,dirs,files in tqdm(os.walk('.\genres', topdown='False')):

        # Iterating through audio files
        for file in files:

            y, sr = librosa.load(os.path.join(root,file), sr=22050)

            feature_list1 = []
            feature_list2 = []
            feature_list3 = []
            feature_list4 = []
            filename = []
            
            # Extract the class name of the song
            for i in file:
              if i == '.':
                  break
              filename.append(i)

            filename = ''.join(filename)

            # Fill the list with the features
            feature_list1.append(filename)
            feature_list2.append(filename)
            feature_list3.append(filename)
            feature_list4.append(filename)
            
            
           
            zcr = librosa.feature.zero_crossing_rate(y + 0.0001, frame_length=2048, hop_length=512)[0]
            
            feature_list1.append(np.std(zcr[0:431]))
            feature_list1.append(np.mean(zcr[0:431]))
            feature_list1.append(np.mean(zcr[0:431])/np.std(zcr[0:431]))
            feature_list1.append(np.max(zcr[0:431])/np.mean(zcr[0:431]))
            feature_list1.append(np.min(zcr[0:431]) / np.mean(zcr[0:431]))
            
            feature_list2.append(np.std(zcr[431:862]))
            feature_list2.append(np.mean(zcr[431:862]))
            feature_list2.append(np.mean(zcr[431:862])/np.std(zcr[431:862]))
            feature_list2.append(np.max(zcr[431:862])/np.mean(zcr[431:862]))
            feature_list2.append(np.min(zcr[431:862]) / np.mean(zcr[431:862]))
            
            feature_list3.append(np.std(zcr[862:1294]))
            feature_list3.append(np.mean(zcr[862:1294]))
            feature_list3.append(np.mean(zcr[862:1294])/np.std(zcr[862:1294]))
            feature_list3.append(np.max(zcr[862:1294])/np.mean(zcr[862:1294]))
            feature_list3.append(np.min(zcr[862:1294]) / np.mean(zcr[862:1294]))
            
    

            rms = librosa.feature.rms(y + 0.0001)[0]
            
            feature_list1.append(np.std(rms[0:431]))
            feature_list1.append(np.mean(rms[0:431]))
            feature_list1.append(np.mean(rms[0:431])/np.std(rms[0:431]))
            feature_list1.append(np.max(rms[0:431])/np.mean(rms[0:431]))
            feature_list1.append(np.min(rms[0:431]) / np.mean(rms[0:431]))
            
            feature_list2.append(np.std(rms[431:862]))
            feature_list2.append(np.mean(rms[431:862]))
            feature_list2.append(np.mean(rms[431:862])/np.std(rms[431:862]))
            feature_list2.append(np.max(rms[431:862])/np.mean(rms[431:862]))
            feature_list2.append(np.min(rms[431:862]) / np.mean(rms[431:862]))
            
            feature_list3.append(np.std(rms[862:1294]))
            feature_list3.append(np.mean(rms[862:1294]))
            feature_list3.append(np.mean(rms[862:1294])/np.std(rms[862:1294]))
            feature_list3.append(np.max(rms[862:1294])/np.mean(rms[862:1294]))
            feature_list3.append(np.min(rms[862:1294]) / np.mean(rms[862:1294]))
            
            
            

            spectral_centroids = librosa.feature.spectral_centroid(y+0.01, sr=sr)[0]
            
            feature_list1.append(np.mean(spectral_centroids[0:431]))
            feature_list1.append(np.std(spectral_centroids[0:431]))    
            feature_list1.append(np.mean(spectral_centroids[0:431])/np.std(spectral_centroids[0:431]))
            feature_list1.append(np.max(spectral_centroids[0:431])/np.mean(spectral_centroids[0:431]))
            feature_list1.append(np.min(spectral_centroids[0:431]) / np.mean(spectral_centroids[0:431]))
            
            feature_list2.append(np.mean(spectral_centroids[431:862]))
            feature_list2.append(np.std(spectral_centroids[431:862]))  
            feature_list2.append(np.mean(spectral_centroids[431:862])/np.std(spectral_centroids[431:862]))
            feature_list2.append(np.max(spectral_centroids[431:862])/np.mean(spectral_centroids[431:862]))
            feature_list2.append(np.min(spectral_centroids[431:862]) / np.mean(spectral_centroids[431:862]))
            
            feature_list3.append(np.mean(spectral_centroids[862:1294]))
            feature_list3.append(np.std(spectral_centroids[862:1294]))   
            feature_list3.append(np.mean(spectral_centroids[862:1294])/np.std(spectral_centroids[862:1294]))
            feature_list3.append(np.max(spectral_centroids[862:1294])/np.mean(spectral_centroids[862:1294]))
            feature_list3.append(np.min(spectral_centroids[862:1294]) / np.mean(spectral_centroids[862:1294]))
            
            

            spectral_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr, roll_percent = 0.85)[0]
            
              
            feature_list1.append(np.mean(spectral_rolloff[0:431]))
            feature_list1.append(np.std(spectral_rolloff[0:431]))    
            feature_list1.append(np.mean(spectral_rolloff[0:431])/np.std(spectral_rolloff[0:431]))
            feature_list1.append(np.max(spectral_rolloff[0:431])/np.mean(spectral_rolloff[0:431]))
            feature_list1.append(np.min(spectral_rolloff[0:431]) / np.mean(spectral_rolloff[0:431]))
            
            feature_list2.append(np.mean(spectral_rolloff[431:862]))
            feature_list2.append(np.std(spectral_rolloff[431:862]))  
            feature_list2.append(np.mean(spectral_rolloff[431:862])/np.std(spectral_rolloff[431:862]))
            feature_list2.append(np.max(spectral_rolloff[431:862])/np.mean(spectral_rolloff[431:862]))
            feature_list2.append(np.min(spectral_rolloff[431:862]) / np.mean(spectral_rolloff[431:862]))
            
            feature_list3.append(np.mean(spectral_rolloff[862:1294]))
            feature_list3.append(np.std(spectral_rolloff[862:1294]))   
            feature_list3.append(np.mean(spectral_rolloff[862:1294])/np.std(spectral_rolloff[862:1294]))
            feature_list3.append(np.max(spectral_rolloff[862:1294])/np.mean(spectral_rolloff[862:1294]))
            feature_list3.append(np.min(spectral_rolloff[862:1294]) / np.mean(spectral_rolloff[862:1294]))
            
            
            
           

            spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y+0.01, sr=sr, p=2)[0]
            
            feature_list1.append(np.mean(spectral_bandwidth_2[0:431]))
            feature_list1.append(np.std(spectral_bandwidth_2[0:431]))    
            feature_list1.append(np.mean(spectral_bandwidth_2[0:431])/np.std(spectral_bandwidth_2[0:431]))
            feature_list1.append(np.max(spectral_bandwidth_2[0:431])/np.mean(spectral_bandwidth_2[0:431]))
            feature_list1.append(np.min(spectral_bandwidth_2[0:431]) / np.mean(spectral_bandwidth_2[0:431]))
            
            feature_list2.append(np.mean(spectral_bandwidth_2[431:862]))
            feature_list2.append(np.std(spectral_bandwidth_2[431:862]))  
            feature_list2.append(np.mean(spectral_bandwidth_2[431:862])/np.std(spectral_bandwidth_2[431:862]))
            feature_list2.append(np.max(spectral_bandwidth_2[431:862])/np.mean(spectral_bandwidth_2[431:862]))
            feature_list2.append(np.min(spectral_bandwidth_2[431:862]) / np.mean(spectral_bandwidth_2[431:862]))
            
            feature_list3.append(np.mean(spectral_bandwidth_2[862:1294]))
            feature_list3.append(np.std(spectral_bandwidth_2[862:1294]))   
            feature_list3.append(np.mean(spectral_bandwidth_2[862:1294])/np.std(spectral_bandwidth_2[862:1294]))
            feature_list3.append(np.max(spectral_bandwidth_2[862:1294])/np.mean(spectral_bandwidth_2[862:1294]))
            feature_list3.append(np.min(spectral_bandwidth_2[862:1294]) / np.mean(spectral_bandwidth_2[862:1294]))

                
            feature_list1[1:] = np.round(feature_list1[1:], decimals=3)
            feature_list2[1:] = np.round(feature_list2[1:], decimals=3)
            feature_list3[1:] = np.round(feature_list3[1:], decimals=3)

      
            df1 = df1.append(pd.DataFrame(feature_list1, index=col_names).transpose(), ignore_index=True)
            df2 = df2.append(pd.DataFrame(feature_list2, index=col_names).transpose(), ignore_index=True)
            df3 = df3.append(pd.DataFrame(feature_list3, index=col_names).transpose(), ignore_index=True)

df1.to_csv('df_features1.csv', index=False)
df2.to_csv('df_features2.csv', index=False)
df3.to_csv('df_features3.csv', index=False)



