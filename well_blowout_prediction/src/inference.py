import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from well_blowout_prediction.models.lstm_model import LSTMModel

# بارگذاری مدل از فایل
def load_model(model_path, model_class, input_size, hidden_layer_size=50):
    model = model_class(input_size, hidden_layer_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# بارگذاری داده‌ها و پیش‌پردازش آن‌ها برای پیش‌بینی
def load_data(input_path, sequence_length=60):
    df = pd.read_csv(input_path)
    features = df[['pressure', 'temperature', 'flow_rate', 'vibration']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    # تبدیل داده‌ها به توالی‌های زمانی
    dataX = []
    for i in range(sequence_length, len(features_scaled)):
        dataX.append(features_scaled[i-sequence_length:i, :])
    
    return np.array(dataX), scaler

# پیش‌بینی فوران با مدل LSTM یا CNN
def predict_blowout(model_path, input_data_path, model_class, sequence_length=60):
    # بارگذاری مدل
    model = load_model(model_path, model_class, input_size=4)
    
    # بارگذاری داده‌ها برای پیش‌بینی
    X, scaler = load_data(input_data_path, sequence_length)
    
    # تبدیل داده‌ها به فرمت PyTorch
    X = torch.tensor(X, dtype=torch.float32)
    
    # پیش‌بینی فوران
    with torch.no_grad():
        predictions = model(X)
    
    print(f"Predicted blowout pressure: {predictions[-1].item()}")

if __name__ == "__main__":
    model_file = './models/lstm_model.pth'  # مدل آموزش‌دیده شده
    input_file = './data/processed/processed_data_with_features.csv'  # داده‌های پردازش شده
    predict_blowout(model_file, input_file, LSTMModel)  # یا CNNModel برای CNN
