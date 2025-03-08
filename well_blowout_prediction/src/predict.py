import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# مسیر models را به sys.path اضافه می‌کنیم تا ایمپورت‌ها درست کار کنند
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

# ایمپورت مدل و توابع مورد نیاز
from lstm_model import LSTMModel
from preprocessing import create_sequences
from data_loader import load_sensor_data

# تنظیمات دستگاه
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_scaler(scaler_path="models/scaler.pkl"):
    """ بارگذاری اسکالر برای نرمال‌سازی داده‌ها در پیش‌بینی """
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("✅ اسکالر بارگذاری شد. داده‌ها نرمال‌سازی می‌شوند.")
        return scaler
    else:
        print("⚠️ اسکالر یافت نشد، داده‌ها بدون تغییر برگردانده می‌شوند.")
        return None

def predict(data, model_path="models/lstm_model.pth", sequence_length=50, threshold=0.5, scaler_path="models/scaler.pkl"):
    """
    بارگیری مدل ذخیره‌شده و انجام پیش‌بینی روی داده‌های جدید.
    """
    # بارگذاری داده‌ها
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("داده‌ها باید یک مسیر فایل یا DataFrame باشند.")

    # اعمال نرمال‌سازی در صورت وجود اسکالر
    scaler = load_scaler(scaler_path)
    if scaler is not None and hasattr(scaler, "transform"):
        df[df.columns] = scaler.transform(df)

    # تبدیل داده‌ها به دنباله‌های ورودی مناسب برای مدل
    sequences, _ = create_sequences(df, sequence_length)
    X = torch.tensor(sequences, dtype=torch.float32).to(device)

    # تعیین ابعاد ورودی مدل
    input_size = X.shape[2]

    # بارگذاری مدل
    model = LSTMModel(input_size=input_size, hidden_layer_size=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # انجام پیش‌بینی
    with torch.no_grad():
        predictions = model(X).cpu().numpy()

    # محاسبه میانگین و واریانس برای بررسی کیفیت مدل
    mean_pred = np.mean(predictions)
    var_pred = np.var(predictions)
    
    print(f"📊 میانگین پیش‌بینی‌ها: {mean_pred:.4f}")
    print(f"📉 واریانس پیش‌بینی‌ها: {var_pred:.4f}")

    # تعیین برچسب خروجی بر اساس آستانه تصمیم‌گیری
    labels = (predictions > threshold).astype(int)

    # ذخیره خروجی‌ها برای بررسی بیشتر
    df_predictions = pd.DataFrame({"Prediction": predictions.flatten(), "Label": labels.flatten()})
    df_predictions.to_csv("predictions.csv", index=False)
    print("✅ پیش‌بینی‌ها در فایل predictions.csv ذخیره شدند.")

    # 📊 نمایش هیستوگرام مقادیر پیش‌بینی‌شده
    plt.hist(predictions, bins=20, edgecolor='black')
    plt.xlabel("مقادیر پیش‌بینی‌شده")
    plt.ylabel("تعداد")
    plt.title("توزیع مقادیر پیش‌بینی‌شده توسط مدل")
    plt.show()

    return predictions, labels

# تست اجرای کد
if __name__ == "__main__":
    data_path = "./data/processed/processed_data.csv"
    df = load_sensor_data(data_path)

    predictions, labels = predict(df)
    print("📊 نمونه‌ای از پیش‌بینی‌های مدل:\n", predictions[:10])
    print("🔹 نمونه‌ای از برچسب‌های تصمیم‌گیری:\n", labels[:10])
    
    print(f"📊 کمترین مقدار پیش‌بینی: {predictions.min():.4f}")
    print(f"📊 بیشترین مقدار پیش‌بینی: {predictions.max():.4f}")
