import yfinance as yf
import pandas as pd

def fetch_data(ticker, period="2y"):
    print(f"Downloading data for: {ticker}...")
    # Fetching historical data
    df = yf.download(ticker, period=period)
    
    df = df[['Close']]
    
    df = df.dropna()
    
    print(f"Successfully downloaded {len(df)} rows of data.")
    return df

if __name__ == "__main__":
    btc_data = fetch_data("BTC-USD")
    print(btc_data.head())