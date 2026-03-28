from ingestion.adapters.akshare_macro import AkshareMacroAdapter
from ingestion.adapters.akshare_minute import AkshareMinuteBarAdapter
from ingestion.adapters.baostock_daily import BaostockDailyAdapter


def build_default_adapters():
    adapters = [
        BaostockDailyAdapter(),
        AkshareMinuteBarAdapter(),
        AkshareMacroAdapter(),
    ]
    return {adapter.adapter_name: adapter for adapter in adapters}
