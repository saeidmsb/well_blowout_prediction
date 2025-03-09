import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import os


DATA_FILE = "C:/Users/Basseri/Desktop/well_blowout_prediction/data/processed/processed_data.csv"


def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["time", "pressure", "temperature", "flow_rate", "vibration"])


df = load_data()


app = dash.Dash(__name__)


app.layout = html.Div([
    html.H1("⛽ Well Monitoring Dashboard", style={'textAlign': 'center'}),
    
    dcc.Interval(
        id='interval-component',
        interval=5000, 
        n_intervals=0
    ),
    
    dcc.Graph(id='pressure-graph'),
    dcc.Graph(id='temperature-graph'),
    dcc.Graph(id='flow-rate-graph'),
    dcc.Graph(id='vibration-graph'),

    html.Div(id='alerts', style={'color': 'red', 'fontSize': 18, 'textAlign': 'center'})
])


@app.callback(
    [Output('pressure-graph', 'figure'),
     Output('temperature-graph', 'figure'),
     Output('flow-rate-graph', 'figure'),
     Output('vibration-graph', 'figure'),
     Output('alerts', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_dashboard(n):
    df = load_data()
    if df.empty:
        return px.line(), px.line(), px.line(), px.line(), "No data available."
    
    pressure_fig = px.line(df, x='time', y='pressure', title='Pressure Over Time')
    temperature_fig = px.line(df, x='time', y='temperature', title='Temperature Over Time')
    flow_rate_fig = px.line(df, x='time', y='flow_rate', title='Flow Rate Over Time')
    vibration_fig = px.line(df, x='time', y='vibration', title='Vibration Over Time')
    

    alerts = []
    thresholds = {"pressure": 50, "temperature": 10, "flow_rate": 80, "vibration": 0.5}
    latest = df.iloc[-1] if not df.empty else None
    
    if latest is not None:
        for sensor, limit in thresholds.items():
            if latest[sensor] > limit:
                alerts.append(f"⚠️ ALERT: {sensor} is above threshold ({latest[sensor]} > {limit})")
    
    return pressure_fig, temperature_fig, flow_rate_fig, vibration_fig, html.Br().join(alerts) if alerts else "✅ All sensors are normal."


if __name__ == '__main__':
    app.run_server(debug=True)


#http://127.0.0.1:8050