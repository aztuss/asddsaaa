from datetime import datetime, timezone

CRYPTO_SYMBOLS = {"BTCUSD","ETHUSD","BNBUSD","ADAUSD","XRPUSD","SOLUSD"}
FOREX_SYMBOLS = {"EURUSD","GBPUSD","USDJPY","USDCHF","AUDUSD","USDCAD","NZDUSD","EURJPY","GBPJPY","EURGBP"}

def is_market_open(symbol, now=None):
    if now is None:
        now = datetime.now(timezone.utc)

    symbol = str(symbol).upper()
    weekday = now.weekday()
    hour = now.hour

    # crypto always open
    if symbol in CRYPTO_SYMBOLS:
        return True

    # forex
    if symbol in FOREX_SYMBOLS:
        if weekday == 5:
            return False
        if weekday == 6:
            return hour >= 22
        if weekday == 4:
            return hour < 22
        return True

    # others
    return weekday < 5
