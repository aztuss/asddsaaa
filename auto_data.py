import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf

DATA_DIR = "data/historical"

SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "NZDUSD": "NZDUSD=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "EURGBP": "EURGBP=X",

    "US100": "^NDX",
    "US500": "^GSPC",
    "US30": "^DJI",
    "DE40": "^GDAXI",
    "UK100": "^FTSE",
    "JP225": "^N225",

    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "BNBUSD": "BNB-USD",
    "ADAUSD": "ADA-USD",
    "XRPUSD": "XRP-USD",
    "SOLUSD": "SOL-USD",

    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "GOOGL": "GOOGL",
    "AMZN": "AMZN",
    "META": "META",
    "NVDA": "NVDA",
    "TSLA": "TSLA",
    "NFLX": "NFLX",
}

TIMEFRAMES = {
    "1h": {"interval": "1h", "period": "730d"},
    "15m": {"interval": "15m", "period": "60d"},
}

def _csv_path(symbol: str, tf: str) -> str:
    return os.path.join(DATA_DIR, f"{symbol}_{tf}.csv")

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)

    if "volume" not in df.columns:
        df["volume"] = 0.0

    needed = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[needed].copy()
    df.dropna(subset=[c for c in ["open", "high", "low", "close"] if c in df.columns], inplace=True)
    return df

def _download(symbol: str, yf_ticker: str, tf: str) -> pd.DataFrame:
    cfg = TIMEFRAMES[tf]
    try:
        df = yf.download(
            yf_ticker,
            period=cfg["period"],
            interval=cfg["interval"],
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        return _clean(df)
    except Exception as e:
        print(f"[DATA] {symbol} {tf} error: {e}")
        return pd.DataFrame()

def ensure_all_data(force: bool = False):
    os.makedirs(DATA_DIR, exist_ok=True)

    for symbol, yf_ticker in SYMBOLS.items():
        for tf in TIMEFRAMES:
            path = _csv_path(symbol, tf)

            if (not force) and os.path.exists(path):
                print(f"[DATA] skip existing: {path}")
                continue

            print(f"[DATA] downloading {symbol} {tf} ...")
            df = _download(symbol, yf_ticker, tf)

            if df.empty:
                print(f"[DATA] empty: {symbol} {tf}")
                continue

            df.to_csv(path)
            print(f"[DATA] saved: {path} rows={len(df)}")

def refresh_all_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    for symbol, yf_ticker in SYMBOLS.items():
        for tf in TIMEFRAMES:
            path = _csv_path(symbol, tf)

            print(f"[DATA] refreshing {symbol} {tf} ...")
            new_df = _download(symbol, yf_ticker, tf)

            if new_df.empty:
                print(f"[DATA] refresh empty: {symbol} {tf}")
                continue

            if os.path.exists(path):
                try:
                    old_df = pd.read_csv(path, index_col=0, parse_dates=True)
                    old_df.index = pd.to_datetime(old_df.index, utc=True)
                    merged = pd.concat([old_df, new_df])
                    merged = merged[~merged.index.duplicated(keep="last")]
                    merged.sort_index(inplace=True)
                    merged.to_csv(path)
                    print(f"[DATA] updated: {path} rows={len(merged)}")
                except Exception as e:
                    print(f"[DATA] merge failed {symbol} {tf}: {e}")
                    new_df.to_csv(path)
            else:
                new_df.to_csv(path)
                print(f"[DATA] created: {path} rows={len(new_df)}")
