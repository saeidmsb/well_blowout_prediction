import os
import pandas as pd
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineering
from src.training import ModelTrainer
from src.inference import Inference
from models.lstm_model import LSTMModel
from models.cnn_model import CNNModel
import json

def load_config(config_path='configs/config.json'):
    """بارگذاری تنظیمات پروژه از فایل config.json"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    
    config = load_config()
    print("Config Loaded: ", config)

    # مسیریابی به داده‌ها
    data_dir = config['data_dir']
    raw_data_path = os.path.join(data_dir, 'raw', 'sensor_data.csv')  # فرض بر این است که داده‌های حسگر در این مسیر هستند

    # بارگذاری داده‌ها
    data_loader = DataLoader(data_dir=data_dir)
    try:
        raw_data = data_loader.load_data(raw_data_path)
        print("Data Loaded Successfully!")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # پیش‌پردازش داده‌ها
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_data(raw_data)
    print("Data Preprocessed Successfully!")

    # مهندسی ویژگی‌ها
    feature_engineering = FeatureEngineering()
    features = feature_engineering.generate_features(processed_data)
    print("Feature Engineering Done!")

    # انتخاب مدل
    if config['model_type'] == 'LSTM':
        model = LSTMModel(config)
    elif config['model_type'] == 'CNN':
        model = CNNModel(config)
    else:
        print("Invalid model type selected.")
        return

    # آموزش مدل
    model_trainer = ModelTrainer(model)
    trained_model = model_trainer.train(features)
    print("Model Trained Successfully!")

    # پیش‌بینی فوران‌ها
    inference = Inference(model)
    predictions = inference.predict(features)
    print("Predictions: ", predictions)

    # ذخیره نتایج یا مدل
    results_dir = config['results_dir']
    trained_model.save(os.path.join(results_dir, 'trained_model.h5'))
    print(f"Trained model saved to {results_dir}")

if __name__ == "__main__":
    main()
