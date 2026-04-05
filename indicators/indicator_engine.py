import pandas as pd

def compute_indicators(df, cfg=None):
    if df is None or df.empty:
        return df

    df = df.copy()

    cols = {c.lower(): c for c in df.columns}

    close_col = cols.get("close")
    high_col = cols.get("high")
    low_col = cols.get("low")
    open_col = cols.get("open")
    volume_col = cols.get("volume")

    if close_col is None:
        return df

    close = df[close_col]

    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["rsi"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    if high_col and low_col:
        high = df[high_col]
        low = df[low_col]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        df["atr"] = tr.rolling(14, min_periods=14).mean()
        df["atr_14"] = df["atr"]

    df["return_1"] = close.pct_change(1)
    df["return_3"] = close.pct_change(3)
    df["return_5"] = close.pct_change(5)
    df["volatility_10"] = df["return_1"].rolling(10, min_periods=10).std()

    return df

def compute_micro_features(df, lookback=15, cfg=None):
    if df is None or df.empty or len(df) < lookback:
        return {}

    df = df.tail(lookback).copy()

    cols = {c.lower(): c for c in df.columns}

    close_col = cols.get("close")
    high_col = cols.get("high")
    low_col = cols.get("low")
    open_col = cols.get("open")
    volume_col = cols.get("volume")

    if close_col is None:
        return {}

    close = df[close_col]
    returns = close.pct_change().fillna(0)

    out = {
        "micro_return": float(returns.mean()),
        "micro_volatility": float(returns.std()),
        "micro_trend": float(close.iloc[-1] - close.iloc[0]),
    }

    if high_col and low_col:
        out["micro_range"] = float(df[high_col].max() - df[low_col].min())

    if open_col:
        body = (close - df[open_col]).abs()
        out["micro_body_mean"] = float(body.mean())

    if volume_col:
        out["micro_volume_mean"] = float(df[volume_col].fillna(0).mean())

    return out
