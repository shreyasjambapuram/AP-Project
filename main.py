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
def main():   
    if st.button("Fetch & Plot Data"):
        lines = []
        file = open("all_tickers.txt", "r")
        for line in file:
            lines.append(line.strip())
        file.close()
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        valid = True
        for ticker in ticker_list:
            if ticker not in lines:
                valid = False
        if not tickers.strip() and not valid:
            st.error("Please enter at least one ticker symbol.")
        else:      
            end_date = datetime.now()
            start_date = end_date - timedelta(days=duration)
            data = yf.download(ticker_list[0], start=start_date, end=end_date)

            st.write(f"Fetching data from **{start_date.date()}** to **{end_date.date()}**...")

            df = yf.download(ticker_list, start=start_date, end=end_date)

            if df.empty:
                st.error("No data found. Check ticker symbols.")
            else:
                fig = go.Figure()

                if isinstance(df['Close'], pd.DataFrame):
                    for t in ticker_list:
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['Close'][t],
                            mode="lines",
                            name=t
                        ))
                else:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Close'],
                        mode="lines",
                        name=ticker_list[0]
                    ))

                fig.update_layout(
                    title="Stock Closing Prices",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_white",
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Next-Day Price Prediction (LSTM)")
                

                df1 = yf.download(ticker_list[0], start=start_date, end=end_date)
            
                next_price = 0
                with st.spinner("Predicting next day's closing price..."):
                    next_price = predict_next_day(df1)
                st.success("Predicted successfully!")
                st.success(f"Predicted next-day closing price for **{ticker_list[0]}**: **${next_price:.2f}**")
main()