import alpaca_trade_api as tradeapi
from decouple import config
api = tradeapi.REST(config('ALPACA_API_KEY'), config('ALPACA_SECRET_KEY'), api_version='v2')
apl = api.polygon.historic_agg_v2('AAPL', 1, 'day', _from='2019-01-01', to='2019-02-01').df