import pandas as pd
import librosa
import csv
import os
import librosa.feature
import numpy as np
from math import sqrt


header = 'filename zcr_mean zcr_std rms_mean rms_std tempo spectral_centroid_mean spectral_centroid_std spectral_rolloff_mean spectral_rolloff_std pectral_bandwidth_2_mean spectral_bandwidth_2_std spectral_bandwidth_3_mean spectral_bandwidth_3_std spectral_bandwidth_4_mean spectral_bandwidth_4_std'
for i in range(1, 7):
    header += f' spectral_contrast_mean{i}'
for i in range(1, 7):
    header += f' spectral_contrast_std{i}'
for i in range(1, 14):
    header += f' mfcc_mean{i}'
for i in range(1, 14):
    header += f' mfcc_std{i}'
for i in range(1, 13):
    header+=f' chroma_stft_mean{i}'
for i in range(1,13):
    header+=f' chroma_stft_std{i}'
for i in range(1, 79):
    header += f' mfcc_cov{i}'

header += ' label'
header = header.split()
file=open('df_features.csv', 'w', newline='')
with file:
    writer=csv.writer(file)
    writer.writerow(header)
genres='blues classical country disco hiphop jazz metal pop reggae rock'.split()
for g in genres:
    for filename in os.listdir(f'./genres/{g}'):
        songname = f'./genres/{g}/{filename}'
        y, sr = librosa.load(songname, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=1024)
        rms = librosa.feature.rms(y+0.0001)[0]
        spec_cent = librosa.feature.spectral_centroid(y+0.01, sr=sr)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spec_bw_2 = librosa.feature.spectral_bandwidth(y + 0.01, sr=sr, p=2)[0]
        spec_bw_3 = librosa.feature.spectral_bandwidth(y + 0.01, sr=sr, p=3)[0]
        spec_bw_4 = librosa.feature.spectral_bandwidth(y + 0.01, sr=sr, p=4)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y, sr=sr, n_bands=6, fmin=200.0)
        rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr, roll_percent = 0.85)[0]
        zcr = librosa.feature.zero_crossing_rate(y+0.0001, frame_length=2048, hop_length=512)[0]
        tempo=librosa.beat.tempo(y=y , sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        x=np.round(np.mean(rms),decimals=3)
        y=np.round(np.std(rms),decimals=3)
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!rms is not rounding
        print(x,y)
        to_append = f' {np.round(np.mean(zcr), decimals=3)} {np.round(np.std(zcr),decimals=3)} {x} {y} ' \
                    f'{np.round(np.mean(tempo),decimals=3)} {np.round(np.mean(spec_cent),decimals=3)} ' \
                    f'{np.round(np.std(spec_cent),decimals=3)} {np.round(np.mean(rolloff),decimals=3)} ' \
                    f'{np.round(np.std(rolloff),decimals=3)} {np.round(np.mean(spec_bw_2),decimals=3)} ' \
                    f'{np.round(np.std(spec_bw_2),decimals=3)} {np.round(np.mean(spec_bw_3),decimals=3)} ' \
                    f'{np.round(np.std(spec_bw_3),decimals=3)} {np.round(np.mean(spec_bw_4),decimals=3)} ' \
                    f'{np.round(np.std(spec_bw_4),decimals=3)} '

        for e in spectral_contrast:
            to_append += f' {np.round(np.mean(e),decimals=3)}'

        for e in spectral_contrast:
            to_append +=f' {np.round(np.std(e),decimals=3)}'
        for e in mfcc:
            to_append += f' {np.round(np.mean(e),decimals=3)}'

        c = np.cov(mfcc)
        for i in range(1, 14):
            to_append += f' {np.round(sqrt(c[i-1 , i-1]),decimals=3)}'

        to_append += f' {np.round(np.mean(chroma_stft),decimals=3)}'
        to_append += f' {np.round(np.std(chroma_stft),decimals=3)}'

        for i in range(1, 13):
            for j in range(2, 14):
                to_append += f' {np.round(c[i-1, j-1],decimals=3)}'


        to_append += f' {g}'

        file = open('df_features.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())



