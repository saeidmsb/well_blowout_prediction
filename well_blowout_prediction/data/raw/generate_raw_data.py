import pandas as pd
import numpy as np
import os


RAW_DATA_PATH = 'data/raw/'
PROCESSED_DATA_PATH = 'data/processed/'

os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


num_samples = 2000


timestamps = pd.date_range('2025-01-01', periods=num_samples, freq='T')


pressure_data = np.random.uniform(1000, 5000, num_samples)  
temperature_data = np.random.uniform(25, 100, num_samples)  
flow_rate_data = np.random.uniform(50, 200, num_samples)   
vibration_data = np.random.uniform(0, 10, num_samples)     

# 
data = pd.DataFrame({
    'timestamp': timestamps,
    'pressure': pressure_data,
    'temperature': temperature_data,
    'flow_rate': flow_rate_data,
    'vibration': vibration_data
})


data.to_csv(os.path.join(RAW_DATA_PATH, 'sensor_data.csv'), index=False)

print("Raw data generated and saved to 'data/raw/sensor_data.csv'.")
