import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd  # مطمئن شو که pandas هم وارد شده باشد

# مسیر models را به sys.path اضافه می‌کنیم تا ایمپورت‌ها درست کار کنند
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

# ایمپورت تابع load_sensor_data
from data_loader import load_sensor_data
from preprocessing import create_sequences
from lstm_model import LSTMModel  # حالا مدل LSTM از پوشه models وارد می‌شود
from cnn_model import CNNModel

# تنظیمات دستگاه (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(data, sequence_length=50, model_type="LSTM", epochs=10, batch_size=32, learning_rate=0.001, hidden_layer_size=64):
    """
    آموزش مدل LSTM یا CNN روی داده‌های ورودی با استفاده از PyTorch.
    """
    # بررسی اینکه آیا data یک رشته (مسیر فایل) است یا یک DataFrame
    if isinstance(data, str):  # اگر داده‌ها یک رشته (مسیر فایل) باشند
        print("داده‌ها از نوع رشته (مسیر فایل) هستند، بارگذاری فایل CSV انجام می‌شود.")
        df = pd.read_csv(data)  # بارگذاری داده‌ها از فایل CSV
    elif isinstance(data, pd.DataFrame):  # اگر داده‌ها از نوع DataFrame باشند
        df = data
    else:
        raise ValueError("داده‌ها باید یک مسیر فایل یا DataFrame باشند.")

    # تبدیل داده‌ها به دنباله‌های زمانی
    sequences, labels = create_sequences(df, sequence_length)

    # تبدیل به Tensor و انتقال به GPU در صورت امکان
    X = torch.tensor(sequences, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.float32).to(device)

    # تقسیم داده‌ها به آموزش و تست
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # تعیین مدل
    input_size = X.shape[2]  # تعداد ویژگی‌ها
    if model_type == "LSTM":
        model = LSTMModel(input_size, hidden_layer_size).to(device)  # hidden_layer_size به مدل منتقل می‌شود
    elif model_type == "CNN":
        model = CNNModel(input_size).to(device)
    else:
        raise ValueError("نوع مدل نامعتبر است. از 'LSTM' یا 'CNN' استفاده کنید.")

    # تنظیمات آموزش
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # شروع آموزش
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)  # انتقال به GPU در صورت امکان
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"✅ Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # ذخیره مدل پس از اتمام آموزش
    model_save_path = "models/lstm_model.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ مدل ذخیره شد: {model_save_path}")

    # **ارزیابی مدل روی داده‌های تست**
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

# تست اجرای کد
if __name__ == "__main__":
    # مسیر داده‌های پردازش‌شده را تنظیم کن
    data_path = "./data/processed/processed_data.csv"
    
    # بارگیری داده‌ها
    df = load_sensor_data(data_path)
    
    # آموزش مدل
    trained_model = train_model(df, model_type="LSTM", epochs=10, hidden_layer_size=128)  # مقدار hidden_layer_size را مشخص کنید
