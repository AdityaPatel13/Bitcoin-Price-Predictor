import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from main import (
    download_and_prepare_data,
    prepare_features_and_target,
    train_and_evaluate_model,
    make_future_predictions
)

st.set_page_config(
    page_title="Bitcoin Price Predictor",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/4/46/Bitcoin.svg",
    layout="wide"
)

st.markdown("""
<style>
    
    .stApp {
        font-family: 'Google Sans', sans-serif;
        background-color: #000000; 
        color: #FFFFFF; 
    }
    
    .stApp, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p, .stApp div {
        font-family: 'Google Sans', sans-serif;
    }

    .stButton>button {
        background-color: #2196F3; 
        color: white;
        border: none;
        border-radius: 24px;
        padding: 10px 24px;
        font-weight: 500;
        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.3);
    }
            
    .stButton>button:hover {
        background-color: #0d82d4;
        color: white;
    }

    .prediction-card {
        position: relative;
        border-radius: 24px; 
        background-color: #121212;
        transition: all 0.3s ease;
        overflow: hidden;
        padding: 5px;
        z-index: 0;
    }

    .prediction-card::before {
        content: '';
        position: absolute;
        top: 0px;
        left: 0px;
        right: 0px;
        bottom: 0px;
        border-radius: 24px inherit;  
        background: linear-gradient(60deg,#4285F4,#EA4335,#FBBC05,#34A853);
        z-index: -1;
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .prediction-card:hover::before {
        opacity: 1;
    }

    .prediction-card:hover {
        border: 0px solid; 
        border-image: linear-gradient(60deg,#4285F4,#EA4335,#FBBC05,#34A853) 1;
        transform: scale(1.05);
    }       

    .card-content {
        position: relative;
        background-color: #000000;
        padding: 24px;
        border-radius: 24px;
        z-index: 1;
    }

    .positive {
        color: #4CAF50; 
    }
            
    .negative {
        color: #F44336; 
    }
            
    .current-price {
        font-size: 28px;
        font-weight: 700;
        color: #FFEB3B; 
    }
            
    .stSpinner > div {
        color: #000000 !important;
    }
            
    .stAlert {
        border-radius: 12px;
        background-color: #000000;
        border: 1px solid #000000;
        color: #FFFFFF;
    }
            
    .stAlert-title, .stAlert-icon {
        color: #FFFFFF !important;
    }
            
    .stAlert-title {
        font-weight: 700;
    }   

    .stSidebar .stAlert {
        background-color: #000000 !important; 
        border: 1px solid #000000 !important; 
        color: #FFFFFF !important;
    }    
</style>
""", unsafe_allow_html=True)

page_icon="https://upload.wikimedia.org/wikipedia/commons/4/46/Bitcoin.svg"

st.title("₿ Bitcoin Price Predictor")
st.markdown("### Predict Bitcoin prices using Machine Learning")

st.sidebar.header("⚙️ Configuration")

period = st.sidebar.selectbox(
    "Data Period",
    ["30d", "60d", "90d", "1y", "2y"],
    index=4
)

prediction_horizon = st.sidebar.slider(
    "Prediction Horizon (hours)",
    min_value=1,
    max_value=24,
    value=5
)

if st.button("Fetch Data and Predict"):
    with st.spinner("Fetching Bitcoin data..."):
        btc_data = download_and_prepare_data(period)
        
        if btc_data is not None:
            st.success(f"Downloaded {len(btc_data)} hours of data")
            
            X, y, features = prepare_features_and_target(btc_data, prediction_horizon)
            
            with st.spinner("Training model..."):
                results, trained_model, X_train, X_test, y_train, y_test = train_and_evaluate_model(X, y)
            
            future_predictions = make_future_predictions(btc_data, trained_model, results, features)
            
            current_price = btc_data['Close'].iloc[-1]
            st.markdown(f"<div class='current-price'>Current Price: ${current_price:.2f}</div>", unsafe_allow_html=True)

            st.subheader("🔮 Future Predictions")
            if future_predictions is not None:
                future_fig = go.Figure()
    
                # Historical Price Trace (unchanged)
                recent_data = btc_data.tail(48)
                future_fig.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_data['Close'],
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='#E0E0E0', width=2)
                ))

                # --- Start of the fix ---
                # Get the last historical data point to bridge the gap
                last_historical_point = btc_data.index[-1]
                last_historical_price = btc_data['Close'].iloc[-1]

                # Create the time series for future predictions
                future_hours = pd.date_range(
                    start=last_historical_point, # Start from the last historical point
                    periods=len(future_predictions) + 1, # Add 1 to include the starting point
                    freq='H'
                )

                # Combine the last historical price with the future predictions
                full_prediction_series = np.insert(future_predictions, 0, last_historical_price)

                # Plot the combined series to bridge the gap
                future_fig.add_trace(go.Scatter(
                    x=future_hours,
                    y=full_prediction_series,
                    mode='lines+markers',
                    name='Future Predictions',
                    line=dict(color='#2196F3', width=2, dash='dot'),
                    marker=dict(size=6)
                ))
                # --- End of the fix ---

                # Current Price Trace (unchanged)
                future_fig.add_trace(go.Scatter(
                    x=[last_historical_point],
                    y=[last_historical_price],
                    mode='markers',
                    name='Current Price',
                    marker=dict(color='#FFEB3B', size=10)
                ))

                future_fig.update_layout(
                    title="Bitcoin Price Forecast",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_dark",
                    height=500,
                    showlegend=True
                )

                st.plotly_chart(future_fig, use_container_width=True)
                
                cols = st.columns(5)
                for i, price in enumerate(future_predictions[-5:]):
                    change = ((price - current_price) / current_price) * 100
                    change_class = "positive" if change >= 0 else "negative"
                    
                    with cols[i % 5]:
                        st.markdown(f"""
                        <div class='prediction-card'>
                            <div class='card-content'>
                                <h4>Hour {i+1}</h4>
                                <h2>${price:.2f}</h2>
                                <p class='{change_class}'>{change:+.2f}%</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("Unable to generate predictions due to insufficient data")
            
            st.subheader("📊 Model Performance")
            performance_data = []
            
            results_single = results['Linear Regression']
            performance_data.append({
                'Model': 'Linear Regression',
                'RMSE': f"${results_single['RMSE']:.2f}",
                'MAE': f"${results_single['MAE']:.2f}",
                'R²': f"{results_single['R2']:.4f}"
            })
            
            performance_df = pd.DataFrame(performance_data)
            st.table(performance_df)
            
            st.subheader("📈 Price Chart with Predictions")
            
            fig = go.Figure()
            
            recent_data = btc_data.tail(200)
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['Close'],
                mode='lines',
                name='Actual Price',
                line=dict(color='#E0E0E0')
            ))
            
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['MA_21'],
                mode='lines',
                name='21-hour MA',
                line=dict(color='#FFEB3B', dash='dash')
            ))
            
            predictions = results['Linear Regression']['predictions']
            fig.add_trace(go.Scatter(
                x=X_test.index,
                y=predictions,
                mode='lines',
                name='Linear Regression Predictions',
                line=dict(color='#2196F3', dash='dot')
            ))
            
            fig.update_layout(
                title="Bitcoin Price with Predictions",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                template="plotly_dark",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.warning("""
            ⚠️ RISK WARNING: This is for educational purposes only!
            Cryptocurrency trading involves significant risk.
            Never invest more than you can afford to lose.
            """)
        else:
            st.error("Failed to download data. Please check your connection and try again.")

st.sidebar.markdown("---")
st.sidebar.info("""
**How it works:**
1. Fetches real-time Bitcoin data from Yahoo Finance
2. Calculates technical indicators
3. Trains a Linear Regression model on historical data
4. Makes price predictions for the next hours
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ for crypto enthusiasts")
