from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from trading_bot.config import AppConfig
from trading_bot.main import _fetch_plan, run_scan
from trading_bot.models import Instrument
from trading_bot.state import AlertState


class ScanRuntimeTests(unittest.TestCase):
    def test_empty_monthly_response_retries_without_marking_success(self) -> None:
        instrument = Instrument(symbol="TEST", label="TEST", asset_name="Test")
        config = SimpleNamespace(
            app=AppConfig(scan_interval_seconds=300, risk_per_trade_pct=.01),
            universe=SimpleNamespace(enabled=False), symbol_lists={},
            patterns={"break_in": SimpleNamespace(enabled=True, settings={"timeframes": ["1mo"]})},
        )
        provider = SimpleNamespace(fetch_buying_power=lambda: 1000, fetch_ohlcv=lambda *_: pd.DataFrame())
        notifier = Mock()
        with TemporaryDirectory() as directory:
            state = AlertState(Path(directory) / "state.json")
            with patch("trading_bot.main.active_universe", return_value=[instrument]):
                self.assertEqual(run_scan(config, state, notifier, provider), [])
            self.assertIsNone(state.last_fetched_at("TEST|1mo"))
            failed_at = state.last_failed_at("TEST|1mo")
            self.assertIsNotNone(failed_at)
            self.assertEqual(_fetch_plan(config, state, [instrument], ["1mo"], failed_at + timedelta(minutes=30)), [])
            self.assertEqual(_fetch_plan(config, state, [instrument], ["1mo"], failed_at + timedelta(hours=1)), [(instrument, "1mo")])
        notifier.send.assert_not_called()

    def test_blocker_summary_is_logged_without_verbose_diagnostics(self) -> None:
        instrument = Instrument(symbol="TEST", label="TEST", asset_name="Test")
        config = SimpleNamespace(
            app=AppConfig(scan_interval_seconds=300, risk_per_trade_pct=.01),
            universe=SimpleNamespace(enabled=False), symbol_lists={},
            patterns={"break_in": SimpleNamespace(enabled=True, settings={"timeframes": ["4h"]})},
        )
        provider = SimpleNamespace(fetch_buying_power=lambda: 1000, fetch_ohlcv=lambda *_: pd.DataFrame({"close": [1.0]}))
        with TemporaryDirectory() as directory:
            state = AlertState(Path(directory) / "state.json")
            with patch("trading_bot.main.active_universe", return_value=[instrument]), \
                 patch("trading_bot.main._near_setup_score", return_value=0), \
                 patch("trading_bot.main.run_pattern", return_value=[]), \
                 patch("trading_bot.main.diagnose_pattern", return_value=["break_in long near-miss; missing=master_candle"]), \
                 patch("trading_bot.main.log") as log:
                run_scan(config, state, Mock(), provider)
        messages = [call.args[0] for call in log.call_args_list]
        self.assertTrue(any("break_in long/master_candle x1" in line for line in messages))
        self.assertFalse(any(line.startswith("Diagnostic break_in TEST") for line in messages))


if __name__ == "__main__":
    unittest.main()
