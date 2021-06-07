assert side.lower() in ['buy', 'sell']
assert notional > 0

data = {
        "symbol": symbol,
        "notional": notional,
        "side": side,
        "type": "market",
        "time_in_force": "day"
    }

try:  
    api._request('POST', '/orders', data=data)
    logging.info(f"Placed order for ${notional} of {symbol}")
except Exception as e:
    logging.error(f"{e} ({symbol})")`
