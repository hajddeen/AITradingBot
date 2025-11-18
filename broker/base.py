# broker/base.py
from abc import ABC, abstractmethod

class BrokerBase(ABC):
    """Common minimal interface for market-data and order operations."""

    @abstractmethod
    def connect(self):
        """Initialize connection to broker / terminal. Raise on failure."""
        pass

    @abstractmethod
    def subscribe_ticks(self, symbol: str, tick_callback):
        """
        Start delivering ticks by calling tick_callback(tick_dict) whenever
        a new tick is available.

        This may be implemented as a background thread, asyncio task, or polling.
        tick_callback must be fast; recommended to use queue.put_nowait inside.
        """
        pass

    @abstractmethod
    def close(self):
        """Cleanly close the connection and stop any background work."""
        pass

    # Optional order methods (not used by the current pipeline but useful later)
    def place_order(self, *args, **kwargs):
        raise NotImplementedError("place_order not implemented for this broker")