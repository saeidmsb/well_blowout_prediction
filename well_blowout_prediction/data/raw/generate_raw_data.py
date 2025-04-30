import pandas as pd
import numpy as np
import os
from sklearn.mixture import GaussianMixture


RAW_DATA_PATH = 'data/raw/'
PROCESSED_DATA_PATH = 'data/processed/'

os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)


num_samples = 2000


def generate_gmm_data(n_samples, means, stds, weights=None):
    if weights is None:
        weights = np.ones(len(means)) / len(means)
    
    n_components = len(means)
    data = []
    
    component_samples = np.random.choice(range(n_components), size=n_samples, p=weights)
    
    for comp in component_samples:
        sample = np.random.normal(loc=means[comp], scale=stds[comp])
        data.append(sample)
        
    return np.array(data)


pressure_data = generate_gmm_data(
    num_samples,
    means=[4000, 10000, 16000],   
    stds=[500, 1000, 1500],
    weights=[0.3, 0.5, 0.2]
)

temperature_data = generate_gmm_data(
    num_samples,
    means=[70, 150, 220],   
    stds=[10, 20, 15],
    weights=[0.4, 0.4, 0.2]
)

vibration_data = generate_gmm_data(
    num_samples,
    means=[0.5, 5, 12],    
    stds=[0.2, 1, 2],
    weights=[0.4, 0.4, 0.2]
)

flow_rate_data = generate_gmm_data(
    num_samples,
    means=[400, 800, 1300],    
    stds=[50, 100, 100],
    weights=[0.3, 0.5, 0.2]
)


data = pd.DataFrame({
    'pressure': pressure_data,
    'temperature': temperature_data,
    'vibration': vibration_data,
    'flow_rate': flow_rate_data
})


a = 0.4
b = 0.2
c = 1.5
d = 0.1


data['y'] = a * data['pressure'] + b * data['temperature'] + c * data['vibration'] + d * data['flow_rate']


data.to_csv(os.path.join(RAW_DATA_PATH, 'sensor_data.csv'), index=False)

print(" Raw data with target 'y' generated and saved to 'data/raw/sensor_data.csv'.")
