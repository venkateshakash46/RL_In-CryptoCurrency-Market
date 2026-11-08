import yfinance as yf
import pandas as pd

coins = [
    "ADA-USD",
    "BCH-USD",
    "BNB-USD",
    "BTC-USD",
    "DASH-USD",
    "EOS-USD",
    "ETH-USD",
    "IOTA-USD",
    "LINK-USD",
    "LTC-USD",
    "TRX-USD",
    "USDT-USD",
    "XLM-USD",
    "XMR-USD",
    "XRP-USD"
]
prices = {}

for coin in coins:
    data = yf.download(
        coin,
        period="2y",
        interval="1d",
        auto_adjust=True,
        progress=False
    )

    prices[coin] = data["Close"].squeeze()

df = pd.DataFrame(prices)
df = df.dropna()

df.to_csv("real_crypto_data.csv")

print(df.tail())
print("Saved real_crypto_data.csv")
