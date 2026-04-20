from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from trading_bot.main import run_scan
from trading_bot.data_provider import IbkrProvider, _apply_resolved_future_identity
from trading_bot.models import Instrument, TradeSignal
from trading_bot.review import AlertReviewStore, review_quality_report
from trading_bot.state import AlertState


class AlertReviewStoreTests(unittest.TestCase):
    def test_resolved_future_identity_includes_contract_and_expiry(self) -> None:
        instrument = Instrument(symbol="KC", label="KC", asset_name="KC", asset_type="future", exchange="NYBOT", currency="USD")
        detail = SimpleNamespace(
            longName="Coffee C",
            contract=SimpleNamespace(localSymbol="KCZ6", exchange="NYBOT", lastTradeDateOrContractMonth="20261218"),
        )

        _apply_resolved_future_identity(instrument, detail)

        self.assertEqual(instrument.asset_name, "Coffee C | contract KCZ6 | NYBOT | expires Dec 2026")

    def test_buying_power_uses_current_base_currency_value(self) -> None:
        account_values = [
            SimpleNamespace(tag="NetLiquidation", currency="BASE", value="999999.99"),
            SimpleNamespace(tag="TotalCashValue", currency="BASE", value="123456.78"),
            SimpleNamespace(tag="AvailableFunds", currency="BASE", value="75000.00"),
            SimpleNamespace(tag="BuyingPower", currency="USD", value="500000.00"),
            SimpleNamespace(tag="BuyingPower", currency="BASE", value="6000000.00"),
        ]
        ib = SimpleNamespace(
            accountValues=lambda: account_values,
            accountSummary=lambda: self.fail("accountSummary should not be requested when streamed updates exist"),
        )
        provider = IbkrProvider()
        provider._connect = lambda: (ib, None)

        buying_power = provider.fetch_buying_power()

        self.assertEqual(buying_power, 6000000.00)

    def test_paper_alias_uses_gateway_profile(self) -> None:
        provider = IbkrProvider(
            account_mode="paper",
            profiles={
                "paper_gateway": {"paper_port": 4002},
                "paper_tws": {"paper_port": 7497},
            },
        )

        self.assertEqual(provider.paper_port, 4002)

    def test_records_evidence_and_persists_manual_review(self) -> None:
        instrument = Instrument(symbol="TEST", label="TEST", asset_name="Test instrument")
        signal = TradeSignal("break_in", "long", "1d", instrument, "2026-01-02T00:00:00+00:00", 100.0, 95.0, 110.0, 2.0, None)
        frame = pd.DataFrame(
            {"open": [99.0, 100.0], "high": [101.0, 111.0], "low": [98.0, 99.0], "close": [100.0, 110.0], "volume": [100.0, 200.0]},
            index=pd.DatetimeIndex(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"]),
        )
        with TemporaryDirectory() as directory:
            store = AlertReviewStore(Path(directory) / "reviews.db")
            store.record(signal, frame)
            store.review(signal.dedupe_key, "rejected", "Insufficient bearish confirmation")
            records = store.list("rejected")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["review_status"], "rejected")
        self.assertEqual(records[0]["evidence"]["candle"]["close"], 110.0)
        self.assertEqual(records[0]["review_note"], "Insufficient bearish confirmation")

    def test_scanner_records_new_signal_before_notification(self) -> None:
        instrument = Instrument(symbol="TEST", label="TEST", asset_name="Test instrument")
        signal = TradeSignal("break_in", "long", "1d", instrument, "2026-01-02T00:00:00+00:00", 100.0, 95.0, 110.0, 2.0, None)
        frame = pd.DataFrame(
            {"open": [99.0, 100.0], "high": [101.0, 111.0], "low": [98.0, 99.0], "close": [100.0, 110.0], "volume": [100.0, 200.0]},
            index=pd.DatetimeIndex(["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"]),
        )
        config = SimpleNamespace(
            app=SimpleNamespace(risk_per_trade_pct=0.01, position_value_pct=0.01),
            universe=SimpleNamespace(enabled=False),
            watchlist=[instrument],
            symbol_lists={},
            patterns={"break_in": SimpleNamespace(name="break_in", enabled=True, settings={"timeframes": ["1d"]})},
        )
        provider = SimpleNamespace(fetch_buying_power=lambda: 100_000.0, fetch_ohlcv=lambda *_: frame)
        notifier = SimpleNamespace(send=lambda _: None, send_photo=lambda *_: None)

        with TemporaryDirectory() as directory:
            store = AlertReviewStore(Path(directory) / "reviews.db")
            state = AlertState(Path(directory) / "state.json")
            with patch("trading_bot.main.run_pattern", return_value=[signal]), patch("trading_bot.main.render_signal_chart", return_value=None):
                emitted = run_scan(config, state, notifier, provider, review_store=store)
            records = store.list()

        self.assertEqual(emitted, [signal])
        self.assertEqual(records[0]["dedupe_key"], signal.dedupe_key)

    def test_quality_report_groups_manual_outcomes_by_signal_attributes(self) -> None:
        records = [
            {
                "review_status": "accepted",
                "signal": {
                    "pattern": "break_in",
                    "timeframe": "1h",
                    "asset_type": "future",
                    "discovery_source": "european_futures_gainers",
                    "direction": "long",
                },
            },
            {
                "review_status": "rejected",
                "signal": {
                    "pattern": "break_in",
                    "timeframe": "1h",
                    "asset_type": "future",
                    "discovery_source": "european_futures_gainers",
                    "direction": "long",
                },
            },
            {
                "review_status": "pending",
                "signal": {
                    "pattern": "bollinger",
                    "timeframe": "4h",
                    "asset_type": "forex",
                    "discovery_source": None,
                    "direction": "short",
                },
            },
        ]

        report = review_quality_report(records)

        self.assertEqual(report["alerts"], 3)
        self.assertEqual(report["assessed"], 2)
        self.assertEqual(report["acceptance_rate"], 0.5)
        groups_by_pattern = {group["pattern"]: group for group in report["groups"]}
        self.assertEqual(groups_by_pattern["break_in"]["acceptance_rate"], 0.5)
        self.assertEqual(groups_by_pattern["bollinger"]["discovery_source"], "unknown")
        self.assertIsNone(groups_by_pattern["bollinger"]["acceptance_rate"])