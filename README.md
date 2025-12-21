# BullBear AI - Stock Prediction Web App

A Flask web application that predicts stock price movements using machine learning. Built with Python, Flask, and scikit-learn.

## Features

- Search for any stock symbol and view current price
- 6-month price chart visualization
- ML-powered price prediction using Random Forest
- Technical indicators: RSI, MACD, SMA, EMA
- Bull/Bear direction prediction with confidence score

## How It Works

The app uses a Random Forest Regressor to predict next-day stock prices. It calculates various technical indicators from historical data:
- Simple Moving Averages (5, 10, 20 day)
- Exponential Moving Averages (12, 26 day)
- Relative Strength Index (RSI)
- MACD
- Volume analysis
- Price volatility

The model is trained on 2 years of historical data and makes predictions based on current market conditions.

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
python BullBearAI.py
```

4. Open http://127.0.0.1:5000 in your browser

## Usage

1. Enter a stock symbol (e.g., AAPL, TSLA, GOOGL)
2. Click Search to view current price and chart
3. Click "Get Prediction" to see the ML prediction

## Tech Stack

- Flask - Web framework
- yfinance - Stock data
- scikit-learn - Machine learning
- pandas/numpy - Data processing
- matplotlib - Chart generation

## Notes

- First prediction for a stock may take longer as the model needs to train
- Trained models are cached in the models/ folder
- Predictions are for educational purposes only

