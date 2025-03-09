import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

def preprocess_data(input_path, output_path, scaler_path):
    
   
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"input'{input_path}'does not find")

   
    df = pd.read_csv(input_path)

    
    df = df.dropna()

    
    scaler = MinMaxScaler()
    columns_to_scale = ['pressure', 'temperature', 'flow_rate', 'vibration']
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  
    df.to_csv(output_path, index=False)

    
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)  
    joblib.dump(scaler, scaler_path)

    print(f" Data processed and saved to '{output_path}'.")
    print(f" Scaler saved to '{scaler_path}'.")

def create_sequences(data, sequence_length=50):
    
    sequences = []
    labels = []

    
    data_array = data[['pressure', 'temperature', 'flow_rate', 'vibration']].values

    for i in range(len(data_array) - sequence_length):
        sequences.append(data_array[i:i + sequence_length])
        labels.append(data_array[i + sequence_length])  

    return np.array(sequences), np.array(labels)

if __name__ == "__main__":
    input_file = './data/raw/sensor_data.csv' 
    output_file = './data/processed/processed_data.csv'  
    scaler_file = './models/scaler.pkl' 

    preprocess_data(input_file, output_file, scaler_file)
