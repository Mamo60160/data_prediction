#!/usr/bin/env python3
##
## EPITECH PROJECT, 2024
## data_prediction
## File description:
## MJ
##

##modèle de random forest pour predire le classement final des pilotesssss

'''
import six
import sys
sys.modules['sklearn.externals.six'] = six
import pydot
import pandas as pd
from id3 import Id3Estimator
from id3 import export_graphviz
from subprocess import check_call
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

np.float = float    
np.int = int 
np.object = object
np.bool = bool
raw_df = pd.read_csv("raw_data.csv")

df = raw_df[["finalRank", "circuitName", "year", "raceRound", "ageAtRace", "seasonAvgPlace", "qualPos", "pointsGoingIn", \
            "winsGoingIn", "recentPlacement", "driverCircuitAvgPlace", "teamCircuitAvgPlace", "positionChange", "seasonOvertake", \
            "careerOvertake"]]

circuit_names = df["circuitName"].unique()
circuit_encode = {}

for idx, circuit in enumerate(circuit_names):
    circuit_encode[circuit] = idx

df['circuitName'] = df['circuitName'].apply(lambda x: circuit_encode[x])

train, test = train_test_split(df, test_size=0.2)

y = train['finalRank'].values

no_target = train.drop(columns = ['finalRank'])
X = no_target.to_numpy()

for row in X:
    row[0] = float(row[0])

clf = RandomForestClassifier(n_estimators = 100, ccp_alpha= 0)
clf.fit(X, y)


def plot_matrix(test_labels, predictions):
    conf_matrix = confusion_matrix(test_labels, predictions)

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_loss(loss, title):
    plt.plot(range(len(loss)), loss)
    plt.title(title)
    plt.show()

test_labels = test['finalRank'].values

no_target_test = test.drop(columns = ['finalRank'])
test_data = no_target_test.to_numpy()

for row in test_data:
    row[0] = float(row[0])

predict_labels = clf.predict(test_data)


acc = accuracy_score(test_labels, predict_labels)
f1 = f1_score(test_labels, predict_labels, average="weighted")

print("Accuracy: ", acc)
print("F-score: ", f1)
plot_matrix(test_labels, predict_labels)

'''