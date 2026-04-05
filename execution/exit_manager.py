def should_close_trade(position, current_price):
    entry = position["entry_price"]
    side = position["side"]

    tp = 0.003   # 0.3%
    sl = 0.002   # 0.2%

    if side == "BUY":
        if current_price >= entry * (1 + tp):
            return True
        if current_price <= entry * (1 - sl):
            return True

    if side == "SELL":
        if current_price <= entry * (1 - tp):
            return True
        if current_price >= entry * (1 + sl):
            return True

    return False
