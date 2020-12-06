import librosa.feature
import csv
import os
import librosa.feature
import numpy as np
from math import sqrt


header = 'tempo chroma_stft_mean chroma_stft_std rms_mean rms_std spectral_centroid_mean spectral_centroid_std spectral_bandwidth_mean spectral_bandwidth_std spectal_bandwidth_2_mean spectal_bandwidth_2_std spectal_bandwidth_3_mean spectal_bandwidth_3_std spectal_bandwidth_4_mean spectal_bandwidth_4_std rolloff_mean rolloff_std zcr_mean zcr_std'
for i in range(1, 7):
    header += f' spectral_contrast_mean{i}'
for i in range(1, 7):
    header += f' spectral_contrast_std{i}'
for i in range(1, 14):
    header += f' mfcc_mean{i}'
for i in range(1, 14):
    header += f' mfcc_std{i}'
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
        to_append = f'{np.mean(tempo)} {np.mean(chroma_stft)} {np.std(chroma_stft)} {np.mean(rms)} {np.std(rms)} {np.mean(spec_cent)} {np.std(spec_cent)} {np.mean(spec_bw)} {np.std(spec_bw)} {np.mean(spec_bw_2)} {np.std(spec_bw_2)} {np.mean(spec_bw_3)} {np.std(spec_bw_3)} {np.mean(spec_bw_4)} {np.std(spec_bw_4)} {np.mean(rolloff)} {np.std(rolloff)} {np.mean(zcr)} {np.std(zcr)}'
        for e in spectral_contrast:
            to_append += f' {np.mean(e)}'
        for e in spectral_contrast:
            to_append +=f' {np.std(e)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        c = np.cov(mfcc)
        for i in range(1, 14):
            to_append += f' {sqrt(c[i-1 , i-1])}'
        for i in range(1, 13):
            for j in range(2, 14):
                to_append += f' {c[i-1, j-1]}'
        to_append+= f' {g}'
        file = open('df_features.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())



