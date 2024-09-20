#!/usr/bin/env python3
##
## EPITECH PROJECT, 2024
## data_prediction
## File description:
## MJ
##
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Modèle de réseau de neurones feedforward
class F1FeedForward(nn.Module):
    def __init__(self):
        super(F1FeedForward, self).__init__()
        self.linear1 = nn.Linear(13, 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, 5)
        self.linear4 = nn.Linear(5, 1)
        self.MSELoss = nn.MSELoss()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x
    
    def train_one_batch(self, attribs, labels, optimizer):
        pred = self(attribs)
        loss = self.MSELoss(pred, labels.unsqueeze(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
    
    # Entraînement sur l'ensemble des données
    def train_all(self, dataloader, optimizer):
        self.train()
        train_loss = [self.train_one_batch(attribs, labels, optimizer) 
                      for attribs, labels in dataloader]
        return sum(train_loss) / len(train_loss)
    
    # Test sur l'ensemble des données
    def test_all(self, dataloader):
        self.eval()
        test_loss = [self.MSELoss(self(attribs), labels.unsqueeze(1)).item() 
                     for attribs, labels in dataloader]
        return sum(test_loss) / len(test_loss)

    # Affichage de la matrice de confusion
    def make_confusion_matrix(self, dataloader):
        self.eval()
        pred_vals = []
        actual_vals = []
        for attribs, labels in dataloader:
            pred = self(attribs)
            pred_vals.append(int(round(pred.item())))
            actual_vals.append(int(labels.item()))
        cm = confusion_matrix(actual_vals, pred_vals)
        st.write("Matrice de confusion :", cm)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues', ax=ax)
        st.pyplot(fig)

# Dataset personnalisé pour les données de F1
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_df: pd.DataFrame):
        self.features = input_df.drop(columns='finalRank').values
        self.labels = input_df['finalRank'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label

# Fonction principale pour prédire les résultats de course
def predict_races():
    st.title("Prédiction des Résultats de Course F1")
    f1_data = pd.read_csv('raw_data.csv')
    st.write("Aperçu des données :")
    st.write(f1_data.head())
    features = ['year', 'circuitId', 'raceRound', 'ageAtRace', 'seasonAvgPlace',
                'qualPos', 'pointsGoingIn', 'winsGoingIn', 'recentPlacement',
                'driverCircuitAvgPlace', 'teamCircuitAvgPlace', 
                'seasonOvertake', 'careerOvertake', 'finalRank']
    f1_data = f1_data[features]
    f1_train, f1_temp = train_test_split(f1_data, test_size=0.2, random_state=42)       #
    f1_test, f1_valid = train_test_split(f1_temp, test_size=0.5, random_state=42)
    st.write(f"Ensemble d'entraînement : {len(f1_train)}")
    st.write(f"Ensemble de test : {len(f1_test)}")
    st.write(f"Ensemble de validation : {len(f1_valid)}")
    train_loader = DataLoader(CustomDataset(f1_train), batch_size=64, shuffle=True)                     # Création des DataLoader pour la gestion des batches
    test_loader = DataLoader(CustomDataset(f1_test), batch_size=1)
    valid_loader = DataLoader(CustomDataset(f1_valid), batch_size=1)
    epochs = st.slider('Nombre d\'équipes', min_value=12, max_value=20, value=20)
    lr = st.slider('Taux d\'apprentissage (LR)', min_value=1e-5, max_value=1e-2, value=1e-3)
    model = F1FeedForward()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1E-6)
    train_loss = []
    valid_loss = []
    test_loss = []
    for epoch in range(epochs):                                                     # Boucles pour entraîner
        train_loss.append(model.train_all(train_loader, optimizer))
        valid_loss.append(model.test_all(valid_loader))
        test_loss.append(model.test_all(test_loader))
    fig, ax = plt.subplots()                                                                 # Affichage des courbes de perte
    ax.plot(train_loss, label="Train Loss")
    ax.plot(valid_loss, label="Validation Loss")
    ax.set_xlabel("Époques")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig)
    model.make_confusion_matrix(test_loader)                    # Affichage de la matrice

if __name__ == "__main__":
    predict_races()
