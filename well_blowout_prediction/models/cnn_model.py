import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes=1):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 30, num_classes)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  
        x = self.fc1(x)
        return x


def load_data(input_path, sequence_length=60):
    df = pd.read_csv(input_path)
    features = df[['pressure', 'temperature', 'flow_rate', 'vibration']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    features_scaled = scaler.fit_transform(features)
    
    
    dataX, dataY = [], []
    for i in range(sequence_length, len(features_scaled)):
        dataX.append(features_scaled[i-sequence_length:i, :])
        dataY.append(features_scaled[i, 0])  
    
    return np.array(dataX), np.array(dataY), scaler

def train_cnn_model(input_path, model_output_path):
    
    X, y, scaler = load_data(input_path)
    

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    
   
    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
   
    model = CNNModel(input_channels=X.shape[2])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    

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
    

    torch.save(model.state_dict(), model_output_path)
    print(f"CNN model trained and saved to {model_output_path}")

if __name__ == "__main__":
    input_file = './data/processed/processed_data_with_features.csv' 
    model_output_file = './models/cnn_model.pth'  
    train_cnn_model(input_file, model_output_file)
