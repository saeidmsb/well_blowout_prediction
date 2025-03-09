import pandas as pd
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from concurrent.futures import ThreadPoolExecutor


CRITICAL_THRESHOLDS = {"pressure": 50, "temperature": 10, "flow_rate": 80, "vibration": 0.5}
COST_FACTORS = {"pressure": 1000, "temperature": 750, "flow_rate": 500, "vibration": 1200}
WELL_DOWNTIME_COST = 1_000_000  
ALERT_THRESHOLD = 3  

logging.basicConfig(filename="alerts.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_password"
RECIPIENT_EMAIL = "recipient_email@gmail.com"

def send_email_alert(subject, message):
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            msg = MIMEMultipart()
            msg["From"] = SENDER_EMAIL
            msg["To"] = RECIPIENT_EMAIL
            msg["Subject"] = subject
            msg.attach(MIMEText(message, "plain"))
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
            print("📧 Email alert sent successfully!")
    except Exception as e:
        print(f"⚠️ Error sending email: {e}")

def load_sensor_data(file_path):
    
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print("⚠️ Data file not found!")
        return None

def calculate_operational_cost(df):
    
    return sum((df[var] > CRITICAL_THRESHOLDS[var]).sum() * COST_FACTORS[var] for var in CRITICAL_THRESHOLDS if var in df.columns)

def calculate_downtime_cost(df):
    
    total_rows = len(df)
    critical_events = sum((df[var] > CRITICAL_THRESHOLDS[var]).sum() for var in CRITICAL_THRESHOLDS if var in df.columns)
    return WELL_DOWNTIME_COST if critical_events / total_rows > 0.05 else 0

def check_real_time_alerts(df):
    
    alert_count = {sensor: 0 for sensor in CRITICAL_THRESHOLDS.keys()}
    
    def process_row(row):
        for sensor, threshold in CRITICAL_THRESHOLDS.items():
            if row[sensor] > threshold:
                alert_count[sensor] += 1
                message = f"⚠️ ALERT: {sensor} value ({row[sensor]}) exceeded the critical threshold {threshold}!"
                logging.warning(message)
                print(message)
                if alert_count[sensor] >= ALERT_THRESHOLD:
                    emergency_message = f"🚨 EMERGENCY! {sensor} exceeded the critical threshold {ALERT_THRESHOLD} times!"
                    logging.error(emergency_message)
                    print(emergency_message)
                    send_email_alert("🚨 Well Emergency Alert!", emergency_message)
            else:
                alert_count[sensor] = 0
    
    with ThreadPoolExecutor() as executor:
        executor.map(process_row, [row for _, row in df.iterrows()])

def main():
    
    file_path = "C:/Users/Basseri/Desktop/well_blowout_prediction/data/processed/processed_data.csv"
    df = load_sensor_data(file_path)
    if df is None:
        return
    
    print("🔍 Analyzing real-time sensor data...")
    with ThreadPoolExecutor() as executor:
        operational_future = executor.submit(calculate_operational_cost, df)
        downtime_future = executor.submit(calculate_downtime_cost, df)
        alert_future = executor.submit(check_real_time_alerts, df)
        
        operational_cost = operational_future.result()
        downtime_cost = downtime_future.result()
        total_cost = operational_cost + downtime_cost
    
    print(f" Operational Cost: ${operational_cost:,.2f}")
    print(f" Downtime Cost: ${downtime_cost:,.2f}")
    print(f" Total Cost: ${total_cost:,.2f}")

if __name__ == "__main__":
    main()
