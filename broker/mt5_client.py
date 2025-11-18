# broker/mt5_client.py
import threading
import time
from datetime import datetime
import MetaTrader5 as mt5
from broker.base import BrokerBase

class MT5Broker(BrokerBase):
    """
    Minimal MT5 broker client that polls MT5 terminal for ticks and invokes a callback.
    This implementation uses a background thread and is simple to integrate.
    """

    def __init__(self, symbol="EURUSD", poll_ms=5):
        self.symbol = symbol
        self.poll_ms = poll_ms
        self._running = False
        self._thread = None

    def connect(self):
        if not mt5.initialize():
            raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
        if not mt5.symbol_select(self.symbol, True):
            raise RuntimeError(f"MT5 symbol_select failed: {self.symbol}")
        # Optional: set symbol properties, ticks etc.
        return True

    def _poll_loop(self, tick_callback):
        last_time_msc = None
        while self._running:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                time.sleep(self.poll_ms / 1000.0)
                continue
            # MT5 returns time_msc (ms since epoch), skip duplicates
            try:
                tm = int(getattr(tick, "time_msc", 0) or 0)
            except Exception:
                tm = 0
            if last_time_msc is None or tm != last_time_msc:
                last_time_msc = tm
                # Build canonical tick dict used by your pipeline
                t = {
                    "bid1_price": float(tick.bid),
                    "ask1_price": float(tick.ask),
                    # MT5 does not provide real per-side sizes; use volume as proxy
                    "bid1_size": float(getattr(tick, "volume", 0) or 0),
                    "ask1_size": float(getattr(tick, "volume", 0) or 0),
                    "trade_flow": float(0.0),
                    "vwap": (float(tick.bid) + float(tick.ask)) / 2.0,
                    "bid2_price": float(tick.bid),
                    "ask2_price": float(tick.ask),
                    "extra_feature": 0.0,
                    "ts": tm
                }
                # callback should be fast (push into asyncio queue)
                try:
                    tick_callback(t)
                except Exception:
                    # swallow to keep polling alive
                    pass
            time.sleep(self.poll_ms / 1000.0)

    def subscribe_ticks(self, symbol: str, tick_callback):
        """Start background polling and call tick_callback for each new tick."""
        # allow symbol override at subscribe time
        if symbol:
            self.symbol = symbol
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._poll_loop, args=(tick_callback,), daemon=True)
            self._thread.start()

    def close(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            mt5.shutdown()
        except Exception:
            pass
