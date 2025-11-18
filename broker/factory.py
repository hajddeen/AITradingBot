# broker/factory.py
from broker.base import BrokerBase

def get_broker(name: str, **kwargs) -> BrokerBase:
    name = name.lower()
    if name in ("mt5", "metaquotes", "metatrader5"):
        from broker.mt5_client import MT5Broker
        return MT5Broker(**kwargs)
    # Add more backends here later (ibkr, binance, custom)
    raise ValueError(f"Unknown broker: {name}")
