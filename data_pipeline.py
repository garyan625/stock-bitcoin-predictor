import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_fetcher import fetch_data

def get_prepared_data(ticker="BTC-USD", time_step=60):
    """
    Fetches raw data, normalizes it, and creates sequences for LSTM training.
    """
    # 1. Fetch raw data
    raw_data = fetch_data(ticker)
    
    # 2. Normalize the data (Scale to 0-1 range)
    # Important: We need this scaler later to 'inverse_transform' predictions back to currency
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(raw_data)

    # 3. Create sequences
    # X: Previous 'time_step' days
    # y: The price on the day immediately following
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

# This block allows you to run this file directly to verify your data
if __name__ == "__main__":
    print("Preparing data pipeline...")
    X, y, scaler = get_prepared_data("BTC-USD")
    
    print(f"Data pipeline complete.")
    print(f"Shape of X (Samples, TimeSteps): {X.shape}")
    print(f"Shape of y (Targets): {y.shape}")