# Bitcoin Price Predictor

A machine learning-based Bitcoin price prediction application with a Streamlit UI.

## 📚 What is Bitcoin?

Bitcoin is a type of digital money (cryptocurrency) that exists only on computers. Think of it like online Monopoly money, but people actually buy and sell things with it! Here's how it works:

- **No Banks**: Unlike regular money that's controlled by banks and governments, Bitcoin is decentralized - meaning no single person or organization controls it.
- **Digital Gold**: Many people think of Bitcoin as "digital gold" because, like gold, there's a limited amount (only 21 million Bitcoins will ever exist).
- **Blockchain**: Bitcoin transactions are recorded on a public ledger called the blockchain - like a giant, shared notebook that everyone can see but no one can erase.
- **Volatility**: Bitcoin's price goes up and down a lot - sometimes by hundreds or thousands of dollars in a single day!

## 🎯 What This Project Does

This project tries to predict Bitcoin's future price using machine learning. It's like having a crystal ball, but instead of magic, it uses math and data!

### How It Works:
1. **Data Collection**: Downloads real-time Bitcoin price data from Yahoo Finance
2. **Analysis**: Calculates technical indicators (like stock trader tools)
3. **Learning**: Uses machine learning to find patterns in the data
4. **Prediction**: Tries to guess where Bitcoin prices might go next
5. **Visualization**: Shows everything in an easy-to-understand web interface

## 🚀 Features

- Real-time Bitcoin price data fetching from Yahoo Finance
- Technical indicators calculation (RSI, Bollinger Bands, Moving Averages)
- Machine learning models for price prediction:
  - Linear Regression
  - Random Forest
  - XGBoost (Extreme Gradient Boosting)
- Interactive Streamlit web interface
- Visualizations of price trends, predictions, and model performance

## 📦 Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## ▶️ Usage

### Command Line Interface
Run the original script:
```
python main.py
```

This will:
1. Download Bitcoin data for the last 60 days
2. Calculate technical indicators
3. Train machine learning models
4. Show predictions for the next 5 hours
5. Display charts with matplotlib

### Web Interface (Streamlit)
Run the Streamlit app:
```
streamlit run app.py
```

This provides a user-friendly web interface where you can:
- Select different time periods (30d, 60d, 90d, 1y)
- Adjust prediction horizon (1-24 hours)
- Choose which technical indicators to display
- Compare different machine learning models

## 🧠 Understanding the Code

### Core Files:
- **main.py**: Contains all the data processing, machine learning, and prediction logic
- **app.py**: Provides the web interface using Streamlit
- **requirements.txt**: Lists all the Python packages needed

### Key Functions in main.py:
1. **download_and_prepare_data()**: Gets Bitcoin data and calculates indicators
2. **calculate_rsi()**: Calculates Relative Strength Index (measures if Bitcoin is overbought/oversold)
3. **calculate_bollinger_bands()**: Creates price bands that show where Bitcoin price typically moves
4. **train_and_evaluate_models()**: Trains machine learning models and compares their performance
5. **make_future_predictions()**: Uses the trained models to predict future prices

### Technical Indicators Explained:
- **Moving Averages (MA)**: Averages of Bitcoin prices over time periods (9, 21, 50 hours)
- **RSI (Relative Strength Index)**: Measures if Bitcoin is overbought (>70) or oversold (<30)
- **Bollinger Bands**: Show price volatility - when price touches upper band, it might go down; when it touches lower band, it might go up
- **Volume**: How much Bitcoin is being traded - high volume often means strong price movements

### Machine Learning Models:
1. **Linear Regression**: Simple model that assumes a linear relationship between features and price
2. **Random Forest**: Ensemble method that combines many decision trees for better accuracy
3. **XGBoost**: Extreme Gradient Boosting - a powerful algorithm that often outperforms others on structured data

## 📊 Understanding the Output

When you run the application, you'll see:

### 1. Future Predictions Cards
- Shows predicted prices for the next few hours
- Green numbers = price going up, Red numbers = price going down
- Percentage change from current price

### 2. Model Performance Table
- **RMSE**: Root Mean Square Error - lower is better (measures average prediction error in dollars)
- **MAE**: Mean Absolute Error - average absolute difference between predicted and actual prices
- **R²**: R-squared score - closer to 1.0 is better (measures how well the model explains price movements)

### 3. Price Chart
- Shows historical Bitcoin prices
- Displays moving averages as trend lines
- Shows Bollinger Bands as upper/lower boundaries
- Plots machine learning predictions

### 4. RSI Chart
- Shows if Bitcoin is overbought (above 70) or oversold (below 30)
- Helps identify potential price reversals

### 5. Feature Importance Chart
- Shows which factors most influence price predictions
- Helps understand what the machine learning model thinks is important

## ⚠️ Important Disclaimer

**This is for educational purposes only!**

- **Not Financial Advice**: These predictions are just computer guesses, not investment advice
- **High Risk**: Cryptocurrency trading involves significant risk
- **No Guarantees**: Machine learning models can be wrong - sometimes very wrong!
- **Never invest more than you can afford to lose**

## 🤔 Why This Is Educational

This project demonstrates:
1. How to collect and process financial data
2. How to calculate technical indicators used by traders
3. How to apply machine learning to time series data
4. How to create interactive data visualizations
5. How to build a complete data science project

Remember: Even if a model has good historical performance, past performance doesn't guarantee future results!

## 🛠️ Configuration Options

In the Streamlit app, you can configure:
- Data period (30d, 60d, 90d, 1y)
- Prediction horizon (1-24 hours)
- Technical indicators to display
- Model selection
