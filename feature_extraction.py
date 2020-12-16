
"""
Feature Extraction
"""

import librosa, librosa.display,librosa.feature
import pandas as pd
import numpy as np
import scipy

from tqdm import tqdm
import os

WAV_DIR = 'genres/'


# Column names of all the features that will be extracted
col_names = ['class','signal_mean','signal_std','signal_skew','signal_kurtosis','zcr_mean',
             'zcr_std', 'rms_mean', 'rms_std','tempo','spectral_flatness_mean',
             'spectral_flatness_std','spectral_centroid_mean',
             'spectral_centroid_std', 'spectral_rolloff_mean', 'spectral_rolloff_std'] + \
            ['spectral_bandwidth_2_mean', 'spectral_bandwidth_2_std',
             'spectral_bandwidth_3_mean', 'spectral_bandwidth_3_std',
             'spectral_bandwidth_4_mean', 'spectral_bandwidth_4_std'] + \
            ['spectral_contrast_' + str(i+1) + '_mean' for i in range(7)] + \
            ['spectral_contrast_' + str(i+1) + '_std' for i in range(7)] + \
            ['mfccs_' + str(i+1) + '_mean' for i in range(13)] + \
            ['mfccs_' + str(i+1) + '_std' for i in range(13)] + \
            ['chroma_stft_' + str(i+1) + '_mean' for i in range(12)] + \
            ['chroma_stft_' + str(i+1) + '_std' for i in range(12)]
      

# Init dataframe to hold the features
df = pd.DataFrame(columns=col_names)

# Parsing throught the file system to extract features from audio
for root,dirs,files in tqdm(os.walk('.\genres', topdown='False')):

        # Iterating through audio files
        for file in files:

            y, sr = librosa.load(os.path.join(root,file), sr=22050)

            feature_list = []
            filename = []
            
            # Extract the class name of the song
            for i in file:
              if i == '.':
                  break
              filename.append(i)

            filename = ''.join(filename)

            # Fill the list with the features
            feature_list.append(filename)
            feature_list.append(np.mean(abs(y)))
            feature_list.append(np.std(y))
            feature_list.append(scipy.stats.skew(abs(y)))
            feature_list.append(scipy.stats.kurtosis(y))
            
            
            zcr = librosa.feature.zero_crossing_rate(y + 0.0001, frame_length=2048, hop_length=512)[0]

            feature_list.append(np.mean(zcr))
            feature_list.append(np.std(zcr))

            rms = librosa.feature.rms(y + 0.0001)[0]
            feature_list.append(np.mean(rms))
            feature_list.append(np.std(rms))

            tempo = librosa.beat.tempo(y, sr=sr)
            feature_list.extend(tempo)
            
            flatness = librosa.feature.spectral_flatness(y=y)
            feature_list.append(np.mean(flatness))
            feature_list.append(np.std(flatness))

            spectral_centroids = librosa.feature.spectral_centroid(y+0.01, sr=sr)[0]
            feature_list.append(np.mean(spectral_centroids))
            feature_list.append(np.std(spectral_centroids))

            spectral_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr, roll_percent = 0.85)[0]
            feature_list.append(np.mean(spectral_rolloff))
            feature_list.append(np.std(spectral_rolloff))

            spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(y+0.01, sr=sr, p=2)[0]
            spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(y+0.01, sr=sr, p=3)[0]
            spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(y+0.01, sr=sr, p=4)[0]
            feature_list.append(np.mean(spectral_bandwidth_2))
            feature_list.append(np.std(spectral_bandwidth_2))
            feature_list.append(np.mean(spectral_bandwidth_3))
            feature_list.append(np.std(spectral_bandwidth_3))
            feature_list.append(np.mean(spectral_bandwidth_4))
            feature_list.append(np.std(spectral_bandwidth_4))

            spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr, n_bands = 6, fmin = 200.0)
            feature_list.extend(np.mean(spectral_contrast, axis=1))
            feature_list.extend(np.std(spectral_contrast, axis=1))

            mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=13)
             
            feature_list.extend(np.mean(mfccs, axis=1))
            feature_list.extend(np.std(mfccs, axis=1))

            chroma_stft = librosa.feature.chroma_stft(y, sr=sr, hop_length=1024)
            feature_list.extend(np.mean(chroma_stft, axis=1))
            feature_list.extend(np.std(chroma_stft, axis=1))
            
         
                
            feature_list[1:] = np.round(feature_list[1:], decimals=3)

      
            df = df.append(pd.DataFrame(feature_list, index=col_names).transpose(), ignore_index=True)

df.to_csv('df_features.csv', index=False)