import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))


from lstm_model import LSTMModel
from preprocessing import create_sequences
from data_loader import load_sensor_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_scaler(scaler_path="models/scaler.pkl"):
    
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded. Data is being normalized.")
        return scaler
    else:
        print("Scaler not found, returning data unchanged.")
        return None

def predict(data, model_path="models/lstm_model.pth", sequence_length=50, threshold=0.5, scaler_path="models/scaler.pkl"):
    
    
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Data must be a DataFrame.")

    #
    scaler = load_scaler(scaler_path)
    if scaler is not None and hasattr(scaler, "transform"):
        df[df.columns] = scaler.transform(df)

    
    sequences, _ = create_sequences(df, sequence_length)
    X = torch.tensor(sequences, dtype=torch.float32).to(device)

    
    input_size = X.shape[2]

    
    model = LSTMModel(input_size=input_size, hidden_layer_size=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    
    with torch.no_grad():
        predictions = model(X).cpu().numpy()

    
    mean_pred = np.mean(predictions)
    var_pred = np.var(predictions)
    
    print(f"📊 Mean prediction: {mean_pred:.4f}")
    print(f"📉 Prediction variance: {var_pred:.4f}")

    # 
    labels = (predictions > threshold).astype(int)

    
    df_predictions = pd.DataFrame({"Prediction": predictions.flatten(), "Label": labels.flatten()})
    df_predictions.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv file.")

    # 
    plt.hist(predictions, bins=20, edgecolor='black')
    plt.xlabel("Predicted values")
    plt.ylabel("Number")
    plt.title("Distribution of predicted values by the model.")
    plt.show()

    return predictions, labels


if __name__ == "__main__":
    data_path = "./data/processed/processed_data.csv"
    df = load_sensor_data(data_path)

    predictions, labels = predict(df)
    print("📊 A sample of model predictions:\n", predictions[:10])
    print("🔹 A sample of decision labels.:\n", labels[:10])
    
    print(f"📊 Minimum predicted value: {predictions.min():.4f}")
    print(f"📊 Maximum predicted value: {predictions.max():.4f}")
