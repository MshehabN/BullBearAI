from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)

# store trained models so we dont have to retrain every time
models_cache = {}
scaler_cache = {}

# want to save models to file so they persist after restart


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search_stock():
    # get the stock symbol from the request
    data = request.json
    symbol = data.get('symbol', '').upper()
    
    if not symbol:
        return jsonify({'error': 'Symbol required'}), 400
    
    try:
        # use yfinance to get stock data
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # try to get current price from info, if not available get from history
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        if not current_price:
            hist = stock.history(period='1d')
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
            else:
                return jsonify({'error': 'Could not get price data'}), 404
        
        # get 6 months of historical data for the chart
        hist = stock.history(period='6mo')
        if hist.empty:
            return jsonify({'error': 'No historical data found'}), 404
        
        # format dates and prices for the frontend
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist['Close'].tolist()
        
        # need to add chart display later
        
        return jsonify({
            'symbol': symbol,
            'price': round(current_price, 2),
            'dates': dates,
            'prices': [round(p, 2) for p in prices]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.json
    symbol = data.get('symbol', '').upper()
    
    if not symbol:
        return jsonify({'error': 'Symbol required'}), 400
    
    try:
        # get 2 years of data for training
        stock = yf.Ticker(symbol)
        hist = stock.history(period='2y')
        
        if hist.empty or len(hist) < 30:
            return jsonify({'error': 'Not enough data'}), 400
        
        # remove any missing values
        df = hist.copy()
        df = df.dropna()
        
        # create technical indicators as features
        # SMA is simple moving average
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        # EMA is exponential moving average
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        # RSI is relative strength index
        df['RSI'] = calculate_rsi(df['Close'], 14)
        # MACD is moving average convergence divergence
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        # volume moving average
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        # high minus low
        df['High_Low'] = df['High'] - df['Low']
        # percentage change in price
        df['Price_Change'] = df['Close'].pct_change()
        # standard deviation as volatility measure
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        df = df.dropna()
        
        if len(df) < 20:
            return jsonify({'error': 'Not enough data after processing'}), 400
        
        # select the features we want to use
        features = ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI', 
                   'MACD', 'Volume_MA', 'High_Low', 'Price_Change', 'Volatility']
        
        # X is features, y is the target (next day price)
        X = df[features].values
        y = df['Close'].shift(-1).values[:-1]
        X = X[:-1]
        
        if len(X) < 20:
            return jsonify({'error': 'Not enough data for training'}), 400
        
        # split into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # scale the features so theyre all on similar scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # create and train the random forest model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # test the model and calculate metrics
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        # calculate accuracy as percentage
        accuracy = 100 - (mae / np.mean(y_test) * 100)
        if accuracy < 0:
            accuracy = 0
        if accuracy > 100:
            accuracy = 100
        
        # store the model so we can use it later
        models_cache[symbol] = model
        scaler_cache[symbol] = scaler
        
        return jsonify({
            'success': True,
            'mae': round(mae, 2),
            'rmse': round(rmse, 2),
            'accuracy': round(accuracy, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    symbol = data.get('symbol', '').upper()
    
    if not symbol:
        return jsonify({'error': 'Symbol required'}), 400
    
    try:
        # if model doesnt exist, train it first
        if symbol not in models_cache:
            stock = yf.Ticker(symbol)
            hist = stock.history(period='2y')
            
            if hist.empty or len(hist) < 30:
                return jsonify({'error': 'Not enough data to train model'}), 400
            
            df = hist.copy()
            df = df.dropna()
            
            # same feature creation as training
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            df['RSI'] = calculate_rsi(df['Close'], 14)
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
            df['High_Low'] = df['High'] - df['Low']
            df['Price_Change'] = df['Close'].pct_change()
            df['Volatility'] = df['Close'].rolling(window=10).std()
            
            df = df.dropna()
            
            if len(df) < 20:
                return jsonify({'error': 'Not enough data after processing'}), 400
            
            features = ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI', 
                       'MACD', 'Volume_MA', 'High_Low', 'Price_Change', 'Volatility']
            
            X = df[features].values
            y = df['Close'].shift(-1).values[:-1]
            X = X[:-1]
            
            if len(X) < 20:
                return jsonify({'error': 'Not enough data for training'}), 400
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            models_cache[symbol] = model
            scaler_cache[symbol] = scaler
        
        # get the trained model
        model = models_cache[symbol]
        scaler = scaler_cache[symbol]
        
        # get recent 3 months of data for prediction
        stock = yf.Ticker(symbol)
        hist = stock.history(period='3mo')
        
        if hist.empty:
            return jsonify({'error': 'No data available'}), 404
        
        # current price is the last closing price
        current_price = hist['Close'].iloc[-1]
        
        # create the same features for prediction
        df = hist.copy()
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        df['RSI'] = calculate_rsi(df['Close'], 14)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        df['High_Low'] = df['High'] - df['Low']
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Close'].rolling(window=10).std()
        
        df = df.dropna()
        
        if df.empty:
            return jsonify({'error': 'Not enough data'}), 400
        
        # get the latest row of features
        features = ['SMA_5', 'SMA_10', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI', 
                   'MACD', 'Volume_MA', 'High_Low', 'Price_Change', 'Volatility']
        X = df[features].iloc[-1].values.reshape(1, -1)
        
        # scale and make prediction
        X_scaled = scaler.transform(X)
        predicted_price = model.predict(X_scaled)[0]
        
        # calculate how much the price will change
        price_change = predicted_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        # determine if its bullish or bearish
        if price_change > 0:
            direction = 'BULL'
        else:
            direction = 'BEAR'
        
        # confidence based on how big the change is
        abs_change = abs(price_change_pct)
        if abs_change < 1:
            confidence = 50
        elif abs_change < 2:
            confidence = 60
        elif abs_change < 3:
            confidence = 70
        elif abs_change < 5:
            confidence = 80
        else:
            confidence = 90
        
        # set colors for the frontend
        if direction == 'BULL':
            direction_color = 'green'
        else:
            direction_color = 'red'
        
        if price_change >= 0:
            change_color = 'green'
        else:
            change_color = 'red'
        
        return jsonify({
            'direction': direction,
            'direction_color': direction_color,
            'current_price': round(current_price, 2),
            'predicted_price': round(predicted_price, 2),
            'price_change': round(price_change, 2),
            'price_change_pct': round(price_change_pct, 2),
            'change_color': change_color,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_rsi(prices, period=14):
    # calculate RSI indicator
    # RSI measures if stock is overbought or oversold
    delta = prices.diff()
    # separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    # calculate relative strength
    rs = gain / loss
    # convert to RSI value between 0-100
    rsi = 100 - (100 / (1 + rs))
    return rsi

# want to add chart visualization later
# want to add watchlist feature later

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

