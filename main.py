import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

def download_and_prepare_data(period="60d"):
    """Download and prepare Bitcoin data with technical indicators"""
    try:
        print("Downloading Bitcoin data...")
        btc_data = yf.download("BTC-USD", period=period, interval="1h")
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    if btc_data.empty:
        print("No data downloaded. Please check connection.")
        return None
    
    if isinstance(btc_data.columns, pd.MultiIndex):
        btc_data.columns = btc_data.columns.droplevel(1)
    
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in btc_data.columns:
            print(f"Missing required column: {col}")
            return None

    print("Calculating technical indicators...")
    
    btc_data['MA_9'] = btc_data['Close'].rolling(window=9).mean()
    btc_data['MA_21'] = btc_data['Close'].rolling(window=21).mean()
    btc_data['MA_50'] = btc_data['Close'].rolling(window=50).mean()
    
    btc_data['Price_Change'] = btc_data['Close'].pct_change()
    btc_data['High_Low_Ratio'] = btc_data['High'] / btc_data['Low'].replace(0, 1e-10)
    btc_data['Volume_MA'] = btc_data['Volume'].rolling(window=20).mean()
    btc_data['Volume_Ratio'] = btc_data['Volume'] / btc_data['Volume_MA'].replace(0, 1e-10)
    
    btc_data['Hour'] = btc_data.index.hour
    btc_data['DayOfWeek'] = btc_data.index.dayofweek
    btc_data['IsWeekend'] = (btc_data['DayOfWeek'] >= 5).astype(int)
    
    btc_data['Volatility'] = btc_data['Close'].rolling(window=24).std()
    
    return btc_data

def prepare_features_and_target(btc_data, prediction_horizon=5):
    """Prepare features and target variable"""
    features = [
        'MA_9', 'MA_21', 'MA_50', 'Price_Change', 'High_Low_Ratio',
        'Volume_Ratio', 'Hour', 'DayOfWeek', 'IsWeekend', 'Volatility'
    ]
    
    btc_data_clean = btc_data.dropna()
    
    X = btc_data_clean[features]
    y = btc_data_clean['Close'].shift(-prediction_horizon)
    
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    return X, y, features

def train_and_evaluate_model(X, y):
    """Train the Linear Regression model and compare performance"""
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining Linear Regression...")
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results = {
        'Linear Regression': {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': mae,
            'R2': r2,
            'predictions': y_pred
        }
    }
    
    trained_model = (model, scaler)
    
    print("Linear Regression Results:")
    print(f"  RMSE: ${np.sqrt(mse):.2f}")
    print(f"  MAE: ${mae:.2f}")
    print(f"  R²: {r2:.4f}")
    
    return results, trained_model, X_train, X_test, y_train, y_test

def make_future_predictions(btc_data, trained_model, results, features, prediction_horizon=5):
    """Make future predictions using the Linear Regression model"""
    model, scaler = trained_model
    
    recent_data = btc_data[features].dropna()
    
    if len(recent_data) < 10:
        recent_data = recent_data.tail(len(recent_data))
    else:
        recent_data = recent_data.tail(10)
    
    if recent_data.empty:
        return None
    
    recent_data_scaled = scaler.transform(recent_data)
    predictions = model.predict(recent_data_scaled)
    
    return predictions

def visualize_results(btc_data, results, X_test, y_test):
    """Create comprehensive visualizations"""
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    ax1 = axes[0]
    recent_data = btc_data.tail(200)
    ax1.plot(recent_data.index, recent_data['Close'], label='Actual Price', color='white', linewidth=1)
    ax1.plot(recent_data.index, recent_data['MA_21'], label='21-hour MA', color='orange', alpha=0.7)
    
    predictions = results['Linear Regression']['predictions']
    ax1.plot(X_test.index, predictions, 
            label='Linear Regression Predictions', linestyle='--', linewidth=2)
    
    ax1.set_title('Bitcoin Price Prediction - Recent History')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (USD)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    predictions = results['Linear Regression']['predictions']
    ax2.scatter(y_test, predictions, alpha=0.6, color='cyan')
    
    min_val, max_val = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Price ($)')
    ax2.set_ylabel('Predicted Price ($)')
    ax2.set_title('Prediction Accuracy - Linear Regression')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    print("=== Bitcoin Price Prediction with Linear Regression ===")
    
    btc_data = download_and_prepare_data("60d")
    if btc_data is None:
        return
    
    print(f"Downloaded {len(btc_data)} hours of data")
    
    X, y, features = prepare_features_and_target(btc_data)
    print(f"Prepared {len(X)} samples with {len(features)} features")
    
    results, trained_model, X_train, X_test, y_train, y_test = train_and_evaluate_model(X, y)
    
    future_predictions = make_future_predictions(btc_data, trained_model, results, features)
    
    print("\n=== Future Predictions (Next 5 hours) - Linear Regression ===")
    current_price = btc_data['Close'].iloc[-1]
    print(f"Current Price: ${current_price:.2f}")
    
    if future_predictions is not None:
        for i, price in enumerate(future_predictions[-5:]):
            change = ((price - current_price) / current_price) * 100
            print(f"Hour {i+1}: ${price:.2f} ({change:+.2f}%)")
    else:
        print("Unable to generate predictions due to insufficient data")
    
    visualize_results(btc_data, results, X_test, y_test)

if __name__ == "__main__":
    main()
