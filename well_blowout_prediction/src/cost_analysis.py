import pandas as pd


CRITICAL_THRESHOLDS = {
    "pressure": 50,       
    "temperature": 10,      
    "flow_rate": 80,       
    "vibration": 0.5        
}


COST_FACTORS = {
    "pressure": 1000,       
    "temperature": 750,      
    "flow_rate": 500,        
    "vibration": 1200       
}

WELL_DOWNTIME_COST = 1_000_000  

def load_sensor_data(file_path):
  
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print("Data file not found.")
        return None

def calculate_operational_cost(df):
    
    if df is None:
        print("Sensor data is invalid.")
        return 0

    total_cost = 0
    for variable, threshold in CRITICAL_THRESHOLDS.items():
        if variable in df.columns:
            exceed_count = (df[variable] > threshold).sum()
            total_cost += exceed_count * COST_FACTORS[variable]

    return total_cost

def calculate_downtime_cost(df):
 
    if df is None:
        return 0

    
    critical_events = 0
    total_rows = len(df)

    for variable, threshold in CRITICAL_THRESHOLDS.items():
        if variable in df.columns:
            critical_events += (df[variable] > threshold).sum()

    
    if critical_events / total_rows > 0.05:
        return WELL_DOWNTIME_COST

    return 0

def main():
   
    file_path = "C:/Users/Basseri/Desktop/well_blowout_prediction/data/processed/processed_data.csv"  
    df = load_sensor_data(file_path)

  
    operational_cost = calculate_operational_cost(df)


    downtime_cost = calculate_downtime_cost(df)


    total_cost = operational_cost + downtime_cost

    print(f"Operational cost based on sensor data: ${operational_cost:,.2f}")
    print(f"Well shutdown cost : ${downtime_cost:,.2f}")
    print(f" Total operational cost: ${total_cost:,.2f}")

if __name__ == "__main__":
    main()
