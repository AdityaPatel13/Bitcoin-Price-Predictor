import unittest
import pandas as pd
from main import (
    download_and_prepare_data,
    prepare_features_and_target
)

class TestBitcoinPrediction(unittest.TestCase):
    
    def test_download_and_prepare_data(self):
        """Test that data can be downloaded and prepared"""
        btc_data = download_and_prepare_data("30d")
        self.assertIsNotNone(btc_data)
        self.assertIsInstance(btc_data, pd.DataFrame)
        self.assertFalse(btc_data.empty)
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_9', 'MA_21', 'MA_50']
        for col in required_columns:
            self.assertIn(col, btc_data.columns)
    
    def test_prepare_features_and_target(self):
        """Test that features and target are prepared correctly"""
        sample_data = {
            'Close': [100, 101, 102, 103, 104, 105],
            'Open': [99, 100, 101, 102, 103, 104],
            'High': [101, 102, 103, 104, 105, 106],
            'Low': [98, 99, 100, 101, 102, 103],
            'Volume': [1000, 1200, 1100, 1300, 1400, 1500]
        }
        df = pd.DataFrame(sample_data, index=pd.date_range(start='2023-01-01', periods=6, freq='H'))
        
        # Simulate prepared data with indicators
        df['MA_9'] = df['Close'].rolling(window=2).mean()
        df['MA_21'] = df['Close'].rolling(window=3).mean()
        df['MA_50'] = df['Close'].rolling(window=4).mean()
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Volume_MA'] = df['Volume'].rolling(window=2).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Hour'] = df.index.hour
        df['DayOfWeek'] = df.index.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        df['Volatility'] = df['Close'].rolling(window=2).std()

        X, y, features = prepare_features_and_target(df, prediction_horizon=1)
        
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), len(y))
        
        # Check that target is shifted correctly
        self.assertEqual(y.iloc[0], 102)

if __name__ == '__main__':
    unittest.main()
