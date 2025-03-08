import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np

def preprocess_data(input_path, output_path, scaler_path):
    """
    پردازش داده‌های خام شامل حذف مقادیر گمشده و نرمال‌سازی.
    """
    # بررسی وجود فایل ورودی
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"فایل ورودی '{input_path}' یافت نشد!")

    # خواندن داده‌های خام از CSV
    df = pd.read_csv(input_path)

    # حذف مقادیر گمشده
    df = df.dropna()

    # نرمال‌سازی داده‌ها
    scaler = MinMaxScaler()
    columns_to_scale = ['pressure', 'temperature', 'flow_rate', 'vibration']
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    # ذخیره داده‌های پردازش‌شده
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # ایجاد پوشه پردازش‌شده (در صورت نیاز)
    df.to_csv(output_path, index=False)

    # ذخیره اسکالر برای استفاده در مرحله پیش‌بینی
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)  # ایجاد پوشه models/ در صورت نیاز
    joblib.dump(scaler, scaler_path)

    print(f"✅ داده‌ها پردازش شدند و در '{output_path}' ذخیره شدند.")
    print(f"✅ اسکالر ذخیره شد در '{scaler_path}'.")

def create_sequences(data, sequence_length=50):
    """
    تبدیل داده‌های سری زمانی به دنباله‌های ورودی برای مدل‌های یادگیری عمیق.
    """
    sequences = []
    labels = []

    # تبدیل داده‌ها به آرایه numpy
    data_array = data[['pressure', 'temperature', 'flow_rate', 'vibration']].values

    for i in range(len(data_array) - sequence_length):
        sequences.append(data_array[i:i + sequence_length])
        labels.append(data_array[i + sequence_length])  # مقدار بعدی به عنوان هدف

    return np.array(sequences), np.array(labels)

if __name__ == "__main__":
    input_file = './data/raw/sensor_data.csv'  # مسیر فایل خام
    output_file = './data/processed/processed_data.csv'  # مسیر ذخیره فایل پردازش‌شده
    scaler_file = './models/scaler.pkl'  # ذخیره اسکالر

    preprocess_data(input_file, output_file, scaler_file)
