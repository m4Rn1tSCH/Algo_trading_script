"""
# Created on: 8/15/2024; 10:37 AM

# Created by: bjaen
"""
"""
alpaca python client available
also contains data access from alpaca
-stock data, options data, crypto currency data
alpaca.data; StockHistoricalDataClient, OptionsData
"""
from datetime import datetime as dt
from datetime import timedelta
from decouple import config
from alpaca.trading.client import TradingClient

from alpaca.data import TimeFrame
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest

from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetAssetsRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetStatus

tc = TradingClient(
    api_key=config('ALPACA_API_KEY'),
    secret_key=config('ALPACA_SECRET_KEY'),
    paper=True
)
tc.get_account()
# dt.today() + timedelta(days=7)
"""
Market Data - input is list
https://alpaca.markets/sdks/python/api_reference/data/stock/requests.html#alpaca.data.requests.StockQuotesRequest
 params get_bars
 symbol: Union[str, List[str]],
 timeframe: TimeFrame,
 start: Optional[str] = None,
 end: Optional[str] = None,
 adjustment: str = 'raw',
 limit: int = None,
 feed: Optional[str] = None,
 asof: Optional[str] = None,
 sort: Optional[Sort] = None
"""


def market_data(symbol_input, time_frame, start, end, limit, sort):
    data_client = StockHistoricalDataClient(
        api_key=config('ALPACA_API_KEY'),
        secret_key=config('ALPACA_SECRET_KEY'))
    sq_req = StockBarsRequest(symbol_or_symbols=symbol_input,
                              timeframe=time_frame,
                              start=start, end=end, limit=limit,
                              sort=sort)
    bars_df = data_client.get_stock_bars(request_params=sq_req).df.tz_convert('America/New_York', level=1)
    return bars_df


# ideally put in one stock symbol
def latest_stock_price(input_list):
    """
    Function for the latest ask_price mostly for limit orders
    :param input_list:
    :return:
    """
    st_c = StockHistoricalDataClient(
        api_key=config('ALPACA_API_KEY'),
        secret_key=config('ALPACA_SECRET_KEY'))
    req_parameters = StockLatestQuoteRequest(symbol_or_symbols=input_list)
    prices = st_c.get_stock_latest_quote(req_parameters)
    v = prices[input_list[0]].ask_price
    return v


# AssetClass.US_EQUITY
def get_tradable_assets(asset_class):
    """
    The asset list is returned as a list of dict-like objects that are directly accessed with x.column
    :param asset_class: pick AssetClass.US_EQUITY
    :return: list of class 'alpaca.trading.models.Asset' ; accessed with .column
    """
    search_eq_tradable = GetAssetsRequest(asset_class=asset_class,
                                          status=AssetStatus.ACTIVE)
    assets = tc.get_all_assets(search_eq_tradable)
    return assets


"""
Retrieve all positions
 returns a list of JSON-elements of all open position in portfolio
 access with element.column
 example of element:
 {   'asset_class': <AssetClass.US_EQUITY: 'us_equity'>,
    'asset_id': UUID('b6d1aa75-5c9c-4353-a305-9e2caa1925ab'),
    'asset_marginable': True,
    'avg_entry_price': '431.18',
    'avg_entry_swap_rate': None,
    'change_today': '-0.0180804090169649',
    'cost_basis': '862.36',
    'current_price': '422.52',
    'exchange': <AssetExchange.NASDAQ: 'NASDAQ'>,
    'lastday_price': '430.3',
    'market_value': '845.04',
    'qty': '2',
    'qty_available': '2',
    'side': <PositionSide.LONG: 'long'>,
    'swap_rate': None,
    'symbol': 'MSFT',
    'unrealized_intraday_pl': '-15.56',
    'unrealized_intraday_plpc': '-0.0180804090169649',
    'unrealized_pl': '-17.32',
    'unrealized_plpc': '-0.0200844194999768',
    'usd': None}
"""
def get_all_positions():
    tc = TradingClient(api_key=config('ALPACA_API_KEY'), secret_key=config('ALPACA_SECRET_KEY'), paper=True)
    pos_l = tc.get_all_positions()
    return pos_l


"""
Market Order submission
 quantity has to be integer
 use OrderSide.BUY or OrderSide.SELL
"""


def submit_market_order(symbol, quantity, order_side):
    mkt_order_data = MarketOrderRequest(
        symbol=symbol,
        qty=quantity,
        side=order_side,
        time_in_force=TimeInForce.DAY
    )
    mo = tc.submit_order(order_data=mkt_order_data)
    return f'order executed for: {mkt_order_data.symbol}'


"""
Limit Order Submissions
"""


def submit_limit_order(symbol, limit_pr, purchase_notional, order_side):
    limit_order_data = LimitOrderRequest(
        symbol=symbol,
        limit_price=limit_pr,
        notional=purchase_notional,
        side=order_side,
        time_in_force=TimeInForce.FOK
    )
    lo = tc.submit_order(order_data=limit_order_data)
    return f"limit order executed for: {limit_order_data.symbol}"


if __name__ == "__main__":
    # list length for stock input can be of length 1 or greater
    # loop for sma tester needs farthest in the past

    # tc = TradingClient(
    #     api_key=config('ALPACA_API_KEY'),
    #     secret_key=config('ALPACA_SECRET_KEY'),
    #     paper=True
    # )
    # tc.get_account()
    # eq = get_tradable_assets(asset_class=AssetClass.US_EQUITY)
    # for i in eq:
    #     print(i.symbol)
    li = get_all_positions()
    stock_list = [st.symbol for st in li]

    a = market_data(symbol_input=['SPY'],
                    time_frame=TimeFrame.Day,
                    start=dt.today() - timedelta(days=365), end=dt.today(),
                    limit=None,
                    sort='desc'
                    )
    # latest_stock_price(input_list=['ALRS'])