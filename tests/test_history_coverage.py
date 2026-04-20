from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from trading_bot.data_provider import BinanceProvider, IbkrProvider
from trading_bot.models import Instrument


class HistoryCoverageTests(unittest.TestCase):
    def test_ibkr_monthly_override_accepts_history_rejected_by_daily_default(self) -> None:
        provider = IbkrProvider(min_rows=220, min_rows_by_interval={"1mo": 100}, request_pause_seconds=0)
        frame = pd.DataFrame({
            "date": pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC"),
            "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "volume": 100.0,
        })
        ib = SimpleNamespace(reqHistoricalData=Mock(return_value=[object()]))
        provider._connect = lambda: (ib, SimpleNamespace(df=lambda _: frame.copy()))
        provider._build_contract = lambda _: SimpleNamespace()
        instrument = Instrument(symbol="TEST", label="TEST", asset_name="Test")
        self.assertEqual(len(provider.fetch_ohlcv(instrument, "1mo")), 120)
        self.assertTrue(provider.fetch_ohlcv(instrument, "1d").empty)

    def test_binance_monthly_override_preserves_default_history_requirement(self) -> None:
        provider = BinanceProvider(min_rows=220, min_rows_by_interval={"1mo": 100})
        dates = pd.date_range("2010-01-01", periods=120, freq="MS", tz="UTC")
        payload = [[int(date.timestamp() * 1000), "10", "12", "9", "11", "100", 0, 0, 0, 0, 0, 0] for date in dates]
        response = Mock()
        response.json.return_value = payload
        instrument = Instrument(symbol="TEST", label="TEST", asset_name="Test")
        with patch("trading_bot.data_provider.requests.get", return_value=response):
            self.assertEqual(len(provider.fetch_ohlcv(instrument, "1mo")), 120)
            self.assertTrue(provider.fetch_ohlcv(instrument, "1d").empty)


if __name__ == "__main__":
    unittest.main()
