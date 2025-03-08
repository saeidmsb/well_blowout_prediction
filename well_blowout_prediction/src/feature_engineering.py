import pandas as pd

def feature_engineering(input_path, output_path):
    
    df = pd.read_csv(input_path)
    
    
    df['pressure_rolling_mean'] = df['pressure'].rolling(window=10).mean()
    df['temperature_rolling_mean'] = df['temperature'].rolling(window=10).mean()
    
    
    df['pressure_diff'] = df['pressure'].diff()
    df['temperature_diff'] = df['temperature'].diff()

    
    df = df.dropna()
    
    
    df.to_csv(output_path, index=False)
    print(f"Feature engineering completed and saved to {output_path}")

if __name__ == "__main__":
    input_file = './data/processed/processed_data.csv' 
    output_file = './data/processed/processed_data_with_features.csv'  
    feature_engineering(input_file, output_file)
