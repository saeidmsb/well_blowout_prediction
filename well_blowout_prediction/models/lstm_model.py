import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.fc(lstm_out[:, -1, :])  # Only take the output of the last time step
        return predictions

# بارگذاری داده‌ها و پیش‌پردازش آن‌ها برای LSTM
def load_data(input_path, sequence_length=60):
    df = pd.read_csv(input_path)
    features = df[['pressure', 'temperature', 'flow_rate', 'vibration']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    # تبدیل داده‌ها به توالی‌های زمانی
    dataX, dataY = [], []
    for i in range(sequence_length, len(features_scaled)):
        dataX.append(features_scaled[i-sequence_length:i, :])
        dataY.append(features_scaled[i, 0])  # پیش‌بینی مقدار فشار (pressure)
    
    return np.array(dataX), np.array(dataY), scaler

def train_lstm_model(input_path, model_output_path):
    # بارگذاری داده‌ها
    X, y, scaler = load_data(input_path)
    
    # تبدیل داده‌ها به فرمت PyTorch
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
    # تعریف DataLoader
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # ساخت مدل LSTM
    model = LSTMModel(input_size=X.shape[2], hidden_layer_size=50)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # آموزش مدل
    model.train()
    for epoch in range(10):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader)}")
    
    # ذخیره مدل
    torch.save(model.state_dict(), model_output_path)
    print(f"LSTM model trained and saved to {model_output_path}")

if __name__ == "__main__":
    input_file = './data/processed/processed_data.csv'  # داده‌های پردازش شده و مهندسی ویژگی
    model_output_file = './models/lstm_model.pth'  # مسیر ذخیره مدل LSTM
    train_lstm_model(input_file, model_output_file)
