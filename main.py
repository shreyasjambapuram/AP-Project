import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim

from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

if "data_df" not in st.session_state:
    st.session_state.data_df = pd.DataFrame()
if "last_fetch_list" not in st.session_state:
    st.session_state.last_fetch_list = []

def fetch_data():
    tickers_input = tickers
    if not tickers_input:
        st.warning("Please enter at least one ticker.")
        return
    ticker_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    end_date = datetime.now()
    start_date = end_date - timedelta(days=duration)
    df = yf.download(ticker_list, start=start_date, end=end_date)
    st.session_state.data_df = df
    st.session_state.last_fetch_list = ticker_list

def plot_data(df, ticker_list):
    st.subheader("Stock Price Chart")
    if isinstance(df.columns, pd.MultiIndex):
        fig = go.Figure()
        for t in ticker_list:
            if t in df['Close'].columns:
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'][t], name=t))
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="Close"))
    st.plotly_chart(fig, use_container_width=True)

def prediction_interface():
    st.sidebar.markdown("---")
    st.sidebar.header("Next‑Day Price Prediction")
    if st.session_state.data_df.empty:
        st.sidebar.info("Fetch data first.")
        return
    target = st.sidebar.selectbox(
        "Select ticker to predict:",
        st.session_state.last_fetch_list
    )
    st.sidebar.write(f"Predicting next day price for **{target}**")
    try:
        if isinstance(st.session_state.data_df.columns, pd.MultiIndex):
            close_series = st.session_state.data_df['Close'][target]
        else:
            close_series = st.session_state.data_df['Close']
        df_temp = pd.DataFrame({"Close": close_series})
        predicted = predict_next_day(df_temp)
        st.sidebar.success(f"Predicted next close: ${predicted:.2f}")
    except Exception as e:
        st.sidebar.error(f"Prediction error: {e}")

def predict_next_day(df):
    data = df["Close"].values.reshape(-1, 1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    seq_length = 30
    x_train = []
    y_train = []

    for i in range(seq_length, len(scaled_data)):
        x_train.append(scaled_data[i-seq_length:i])
        y_train.append(scaled_data[i])

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train_torch = torch.tensor(x_train, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32)

    model = LSTMModel()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 20
    for _ in range(epochs):
        for i in range(len(x_train_torch)):
            seq = x_train_torch[i]
            label = y_train_torch[i]

            optimizer.zero_grad()
            model.hidden_cell = (
                torch.zeros(1, 1, model.hidden_layer_size),
                torch.zeros(1, 1, model.hidden_layer_size)
            )

            prediction = model(seq)
            loss = loss_fn(prediction, label)
            loss.backward()
            optimizer.step()

    last_30 = scaled_data[-30:]
    last_30_tensor = torch.tensor(last_30, dtype=torch.float32)

    model.hidden_cell = (
        torch.zeros(1, 1, model.hidden_layer_size),
        torch.zeros(1, 1, model.hidden_layer_size)
    )

    next_scaled = model(last_30_tensor).detach().numpy()
    next_price = scaler.inverse_transform(next_scaled.reshape(-1, 1))[0][0]

    return next_price


st.set_page_config(page_title="Stock Viewer", layout="wide")

st.title("Interactive Stock Price Viewer")

tickers = st.text_input(
    "Enter stock tickers (comma-separated):",
    placeholder="AAPL, MSFT, TSLA"
)

duration = st.number_input(
    "How many days back should the chart show?",
    min_value=1,
    max_value=3650,
    value=365,
    step=1
)

fetch_button = st.button("Fetch & Plot Data")

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

def check_stock_health(price_list, threshold):
    total_price = 0
    
    for price in price_list:
        total_price = total_price + price
        
    if len(price_list) == 0:
        return "No Data"
    
    average_price = total_price / len(price_list)
    
    if average_price > threshold:
        return f"STRONG ({average_price:.2f} > {threshold})"
    else:
        return f"WEAK ({average_price:.2f} <= {threshold})"

def main():   
    if fetch_button:
        fetch_data()
    
    if not st.session_state.data_df.empty:
        plot_data(st.session_state.data_df, st.session_state.last_fetch_list)
        
        st.sidebar.header("Simple Analysis")
        
        target_ticker = st.session_state.last_fetch_list[0]
        
        if isinstance(st.session_state.data_df.columns, pd.MultiIndex):
            clean_price_list = st.session_state.data_df['Close'][target_ticker].dropna().tolist()
        else:
            clean_price_list = st.session_state.data_df['Close'].dropna().tolist()
            
        user_threshold = st.sidebar.number_input(
            f"Set Target Price for {target_ticker}:", 
            value=100.0,
            step=5.0
        )
        
        status_result = check_stock_health(clean_price_list, user_threshold)
        
        st.sidebar.success(f"{target_ticker} Status: {status_result}")

    prediction_interface()

main()
