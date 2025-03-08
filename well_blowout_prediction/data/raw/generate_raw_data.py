import pandas as pd
import numpy as np
import os

# ایجاد پوشه‌های raw و processed در صورتی که وجود ندارند
RAW_DATA_PATH = 'data/raw/'
PROCESSED_DATA_PATH = 'data/processed/'

os.makedirs(RAW_DATA_PATH, exist_ok=True)
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

# تعداد داده‌ها
num_samples = 2000

# زمان (timestamp) به‌عنوان نمونه داده
timestamps = pd.date_range('2025-01-01', periods=num_samples, freq='T')

# تولید داده‌های تصادفی برای فشار، دما، نرخ جریان و ارتعاش
pressure_data = np.random.uniform(1000, 5000, num_samples)  # فشار بین 1000 تا 5000
temperature_data = np.random.uniform(25, 100, num_samples)  # دما بین 25 تا 100 درجه سانتی‌گراد
flow_rate_data = np.random.uniform(50, 200, num_samples)   # نرخ جریان بین 50 تا 200
vibration_data = np.random.uniform(0, 10, num_samples)     # ارتعاش بین 0 تا 10

# ساخت دیتافریم برای ذخیره داده‌ها
data = pd.DataFrame({
    'timestamp': timestamps,
    'pressure': pressure_data,
    'temperature': temperature_data,
    'flow_rate': flow_rate_data,
    'vibration': vibration_data
})

# ذخیره داده‌ها به‌صورت CSV در پوشه raw
data.to_csv(os.path.join(RAW_DATA_PATH, 'sensor_data.csv'), index=False)

print("Raw data generated and saved to 'data/raw/sensor_data.csv'.")
