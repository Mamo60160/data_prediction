#!/usr/bin/env python3
##
## EPITECH PROJECT, 2024
## data_prediction
## File description:
## MJ
##

''
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

pd.set_option('display.max_columns', None)

results_df = pd.read_csv('data/results.csv')

# merge constructor name/fusionner
constructors_df = pd.read_csv('data/constructors.csv')
merge1_df = pd.merge(results_df, constructors_df[['constructorId', 'constructorRef']], on='constructorId')
merge1_df.rename(columns={'constructorRef': 'constructorName'}, inplace=True)

# placer constructorName juste après constructorId
column_order = merge1_df.columns.tolist()
constructor_index = column_order.index('constructorId')
column_order.insert(constructor_index + 1, column_order.pop(column_order.index('constructorName')))
merge1_df = merge1_df[column_order]

# Fusionner les circuits de course, les années et les manches
races_df = pd.read_csv('data/races.csv')
merge2_pt1_df = pd.merge(merge1_df, races_df[['raceId', 'year', 'round', 'circuitId', 'date']], on='raceId')

# maintenant saisir le circuit en fonction de circuitId
circuits_df = pd.read_csv('data/circuits.csv')
merge2_df = pd.merge(merge2_pt1_df, circuits_df[['circuitId', 'circuitRef']], on='circuitId')
merge2_df.rename(columns={'circuitRef': 'circuitName'}, inplace=True)

# placer toutes les informations après raceId, laisser tomber circuitId
column_order = merge2_df.columns.tolist()
constructor_index = column_order.index('raceId')
column_order.insert(constructor_index + 1, column_order.pop(column_order.index('circuitId')))
column_order.insert(constructor_index + 2, column_order.pop(column_order.index('circuitName')))
column_order.insert(constructor_index+3, column_order.pop(column_order.index('year')))
column_order.insert(constructor_index+4, column_order.pop(column_order.index('date')))
column_order.insert(constructor_index+5, column_order.pop(column_order.index('round')))
merge2_df = merge2_df[column_order]

# fusionner le nom du conducteur
drivers_df = pd.read_csv('data/drivers.csv')
merge3_df = pd.merge(merge2_df, drivers_df[['driverId', 'driverRef']], on='driverId')
merge3_df.rename(columns={'driverRef': 'driverName'}, inplace=True)

# placer driverName juste après driverId
column_order = merge3_df.columns.tolist()
constructor_index = column_order.index('driverId')
column_order.insert(constructor_index + 1, column_order.pop(column_order.index('driverName')))
merge3_df = merge3_df[column_order]

# juste un travail de renommage pour plus de clarté
merge3_df.rename(columns={'date':'raceDate', 'round':'raceRound', 'number':'carNumber', 'grid':'startPos'}, inplace=True)
merge3_df.rename(columns={'position':'finishPos', 'positionOrder':'finalRank', 'rank':'fastestLapPos'}, inplace=True)

# ajouter un statut à partir de statusId
status_df = pd.read_csv('data/status.csv')
merge4_df = pd.merge(merge3_df, status_df[['statusId', 'status']], on='statusId')
merge4_df.rename(columns={'status': 'finishStatus'}, inplace=True)
column_order = merge4_df.columns.tolist()
constructor_index = column_order.index('statusId')
column_order.insert(constructor_index + 1, column_order.pop(column_order.index('finishStatus')))
merge4_df = merge4_df[column_order]

# ajouter des données de qualification
qualifying_df = pd.read_csv('data/qualifying.csv')
merge5_df = pd.merge(merge4_df, qualifying_df[['raceId', 'driverId', 'constructorId', 'position', 'q1', 'q2', 'q3']], on=['raceId', 'driverId', 'constructorId'])
merge5_df.rename(columns={'position': 'qualPos', 'q1': 'q1Time', 'q2': 'q2Time', 'q3': 'q3Time'}, inplace=True)
