import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_sensor_data(file_path, scaler_path=None):
    """
    بارگذاری داده‌های پردازش‌شده و نرمال‌سازی (در صورت نیاز).
    """
    df = pd.read_csv(file_path)

    # بررسی ستون‌های مورد نیاز
    required_columns = ["pressure", "temperature", "flow_rate", "vibration"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"ستون '{col}' در داده‌ها یافت نشد!")

    # بارگذاری اسکالر از فایل
    if scaler_path and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        df[required_columns] = scaler.transform(df[required_columns])
    else:
        print("⚠️ اسکالر یافت نشد، داده‌ها بدون تغییر برگردانده می‌شوند.")

    return df

def create_sequences(data, sequence_length=50):
    """
    تبدیل داده‌های سری زمانی به دنباله‌های مورد نیاز برای LSTM/CNN.
    """
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

    print(f"✅ داده‌های بارگذاری و پردازش شدند. تعداد دنباله‌ها: {len(sequences)}")
