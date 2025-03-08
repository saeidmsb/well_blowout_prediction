import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_sensor_data(file_path, scaler_path=None):
   
    df = pd.read_csv(file_path)

    
    required_columns = ["pressure", "temperature", "flow_rate", "vibration"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}")

    
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        df[required_columns] = scaler.transform(df[required_columns])
    else:
        print("Scalar not found, returning data unchanged.")

    return df

def create_sequences(data, sequence_length=50):
    
    sequences = []
    labels = []

    for i in range(len(data) - sequence_length):
        sequences.append(data.iloc[i:i + sequence_length].values)
        labels.append(data.iloc[i + sequence_length].values)

    return np.array(sequences), np.array(labels)

if __name__ == "__main__":
    file_path = './data/processed/processed_data.csv'
    scaler_path = './models/scaler.pkl'

    df = load_sensor_data(file_path, scaler_path)
    sequences, labels = create_sequences(df, sequence_length=50)

    print(f"Data loaded and processed successfully. Number of sequences: {len(sequences)}")
