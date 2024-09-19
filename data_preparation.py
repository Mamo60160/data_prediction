import pandas as pd
import os

def load_data():
    # Vérifier le répertoire courant
    print(f"Répertoire courant : {os.getcwd()}")
    
    # Vérifier l'existence des fichiers
    assert os.path.exists('data/drivers.csv'), 
    assert os.path.exists('data/races.csv'),
    assert os.path.exists('data/results.csv'),
    assert os.path.exists('data/weather.csv'),
    
    drivers = pd.read_csv('data/drivers.csv')
    races = pd.read_csv('data/races.csv')
    results = pd.read_csv('data/results.csv')
    weather = pd.read_csv('data/weather.csv')
    
    return drivers, races, results, weather

def prepare_data(drivers, races, results, weather):
    # Fusionner les résultats avec les courses
    race_results = pd.merge(results, races, on='raceId')
    # Fusionner avec les pilotes
    race_results = pd.merge(race_results, drivers, on='driverId')
    # Fusionner avec les données météorologiques
    final_df = pd.merge(race_results, weather, on='raceId')
    
    # Nettoyage des données
    final_df = final_df.dropna()
    final_df = final_df.drop_duplicates()
    
    return final_df

def main():
    drivers, races, results, weather = load_data()
    final_df = prepare_data(drivers, races, results, weather)
    final_df.to_csv('data/final_df.csv', index=False)

if __name__ == "__main__":
    main()
