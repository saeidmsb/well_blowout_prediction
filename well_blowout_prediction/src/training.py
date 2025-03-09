import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd  

# 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))


from data_loader import load_sensor_data
from preprocessing import create_sequences
from lstm_model import LSTMModel  
from cnn_model import CNNModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(data, sequence_length=50, model_type="LSTM", epochs=10, batch_size=32, learning_rate=0.001, hidden_layer_size=64):
    
    
    if isinstance(data, str):  
        print("Data is of string type , loading CSV file.")
        df = pd.read_csv(data)  
    elif isinstance(data, pd.DataFrame):  
        df = data
    else:
        raise ValueError("Data must be a DataFrame.")

    # 
    sequences, labels = create_sequences(df, sequence_length)

    
    X = torch.tensor(sequences, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.float32).to(device)

    
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

   
    input_size = X.shape[2] 
    if model_type == "LSTM":
        model = LSTMModel(input_size, hidden_layer_size).to(device)  
    elif model_type == "CNN":
        model = CNNModel(input_size).to(device)
    else:
        raise ValueError("Invalid model type. Use 'LSTM' or 'CNN'.")

    #
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device) 
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f" Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    
    model_save_path = "models/lstm_model.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f" models saved : {model_save_path}")

    #
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_test_loss += loss.item()

    print(f"📊 Test Loss: {total_test_loss / len(test_loader):.4f}")

    return model


if __name__ == "__main__":
    
    data_path = "./data/processed/processed_data.csv"
    
  
    df = load_sensor_data(data_path)
    
    
    trained_model = train_model(df, model_type="LSTM", epochs=10, hidden_layer_size=128) 
