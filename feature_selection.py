"""
Correlated filter methods.
Features should be highly corellated with the target.
Features should be uncorellated among themselves.
We will use the Pearson corellation(default).
"""

#####**** kendall or spearman ****#####

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

train = pd.read_csv('df_features.csv', header=0)
y = train.iloc[0:,0]
x = train.iloc[0:,1:]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state = 2020, stratify=y)

#Create a set to hold the correlated features
corr_features = set()

#Create the correlation matrix
corr_matrix = X_train.corr() #pearson, kendall or spearman

#display a heatmap of the correlation matrix
plt.figure(figsize=(18,8))
sns.heatmap(corr_matrix)


for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i,j]) > 0.9 and (i != j):

            colname1 = corr_matrix.columns[i]
            colname2 = corr_matrix.columns[j]

            corr_features.add(colname1)
            
            corr_features.add(colname2)

# Correlation result
# corr_features = {'mfccs_2_mean', 'spectral_centroid_mean', 'spectral_rolloff_mean'}
