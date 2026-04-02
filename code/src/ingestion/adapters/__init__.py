from ingestion.adapters.base import BaseAdapter
from ingestion.adapters.baostock_daily import BaoStockDailyAdapter
from ingestion.adapters.akshare_macro import AkshareMacroAdapter
from ingestion.adapters.akshare_minute import AkshareMinuteAdapter
from ingestion.adapters.akshare_minute import AkshareMinuteBarAdapter

__all__ = [
    'BaseAdapter',
    'BaoStockDailyAdapter',
    'AkshareMacroAdapter',
    'AkshareMinuteAdapter',
    'AkshareMinuteBarAdapter',
]
