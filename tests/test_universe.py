from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import pandas as pd

from trading_bot.models import Instrument
from trading_bot.universe import UniverseStore, _multi_timeframe_trend_status, _trend_status, _watchlist_candidates, refresh_universe, universe_instrument_payload
from trading_bot.data_provider import IbkrProvider


class UniverseStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instrument = Instrument(
            symbol="TEST",
            label="TEST",
            asset_name="Test instrument",
            provider="ibkr",
            provider_symbol="TEST",
            asset_type="stock",
            exchange="SMART",
            currency="USD",
        )

    def test_load_fresh_returns_members_before_snapshot_expiry(self) -> None:
        with TemporaryDirectory() as directory:
            store = UniverseStore(Path(directory) / "universe.db")
            store.replace([self.instrument])

            members = store.load_fresh(60, datetime.now(timezone.utc))

        self.assertEqual([member.symbol for member in members], ["TEST"])

    def test_writes_json_snapshot_alongside_database(self) -> None:
        with TemporaryDirectory() as directory:
            store = UniverseStore(Path(directory) / "universe.db")
            store.write_json_snapshot([self.instrument])
            payload = (Path(directory) / "universe.json").read_text(encoding="utf-8")

        self.assertIn('"symbol": "TEST"', payload)
        self.assertIn('"source": "core"', payload)

    def test_load_fresh_returns_no_members_after_snapshot_expiry(self) -> None:
        with TemporaryDirectory() as directory:
            store = UniverseStore(Path(directory) / "universe.db")
            store.replace([self.instrument])
            expiry_time = datetime.now(timezone.utc) + timedelta(seconds=61)

            members = store.load_fresh(60, expiry_time)

        self.assertEqual(members, [])

    def test_failed_discovery_retains_existing_dynamic_snapshot(self) -> None:
        core = Instrument(symbol="CORE", label="CORE", asset_name="Core", provider="ibkr", provider_symbol="CORE", asset_type="stock", exchange="SMART", currency="USD")
        dynamic = Instrument(symbol="DYNAMIC", label="DYNAMIC", asset_name="Dynamic", provider="ibkr", provider_symbol="DYNAMIC", asset_type="stock", exchange="SMART", currency="USD")
        with TemporaryDirectory() as directory:
            database_path = Path(directory) / "universe.db"
            UniverseStore(database_path).replace([core, dynamic])
            config = SimpleNamespace(
                watchlist=[core],
                universe=SimpleNamespace(
                    enabled=True,
                    core_symbols=["CORE"],
                    database_path=str(database_path),
                    max_dynamic_candidates=20,
                    sources=[{"name": "most_active", "enabled": True}],
                ),
            )
            provider = SimpleNamespace(scan_instruments=lambda _: (_ for _ in ()).throw(RuntimeError("Socket disconnect")))

            result = refresh_universe(config, provider)

            self.assertTrue(result.retained_previous_snapshot)
            self.assertEqual([instrument.symbol for instrument in result.instruments], ["CORE", "DYNAMIC"])
            self.assertEqual([instrument.symbol for instrument in UniverseStore(database_path).load()], ["CORE", "DYNAMIC"])

    def test_instrument_payload_exposes_discovery_ranking(self) -> None:
        instrument = Instrument(
            symbol="HOT",
            label="HOT",
            asset_name="Hot instrument",
            asset_type="future",
            exchange="CME",
            currency="USD",
            discovery_source="us_futures_gainers",
            discovery_priority=15,
            discovery_rank=2,
        )

        payload = universe_instrument_payload(instrument)

        self.assertEqual(payload["source"], "us_futures_gainers")
        self.assertEqual(payload["scanner_rank"], 2)
        self.assertEqual(payload["asset_type"], "future")

    def test_stock_scanner_excludes_non_common_products_when_configured(self) -> None:
        contract = SimpleNamespace(symbol="SPY", secType="STK", exchange="SMART", primaryExchange="ARCA", currency="USD")
        result = SimpleNamespace(rank=0, contractDetails=SimpleNamespace(contract=contract, stockType="ETF", longName="SPDR S&P 500 ETF Trust"))
        ib = SimpleNamespace(reqScannerData=lambda _: [result])
        provider = IbkrProvider(request_pause_seconds=0)
        provider._connect = lambda: (ib, None)

        instruments = provider.scan_instruments({"allowed_stock_types": ["COMMON"]})

        self.assertEqual(instruments, [])

    def test_trend_status_accepts_aligned_bullish_and_bearish_daily_trends(self) -> None:
        bullish_frame = pd.DataFrame({"close": [100 + index for index in range(80)]})
        bearish_frame = pd.DataFrame({"close": [180 - index for index in range(80)]})
        neutral_frame = pd.DataFrame({"close": [100.0] * 80})

        bullish = _trend_status(bullish_frame, 20, 50, 5, 0.01)
        bearish = _trend_status(bearish_frame, 20, 50, 5, 0.01)
        neutral = _trend_status(neutral_frame, 20, 50, 5, 0.01)

        self.assertEqual(bullish[0] if bullish else None, "bullish")
        self.assertEqual(bearish[0] if bearish else None, "bearish")
        self.assertIsNone(neutral)

    def test_multi_timeframe_trend_requires_weekly_and_monthly_agreement(self) -> None:
        dates = pd.date_range("2020-01-01", periods=1_500, freq="B", tz="UTC")
        bullish_frame = pd.DataFrame({"close": [100 * 1.003**index for index in range(len(dates))]}, index=dates)
        bearish_frame = pd.DataFrame({"close": [10_000 * 0.997**index for index in range(len(dates))]}, index=dates)

        bullish = _multi_timeframe_trend_status(bullish_frame, 20, 50, 5, 0.01)
        bearish = _multi_timeframe_trend_status(bearish_frame, 20, 50, 5, 0.01)

        self.assertEqual(bullish, ("bullish", bullish[1], "aligned"))
        self.assertEqual(bearish, ("bearish", bearish[1], "aligned"))

    def test_multi_timeframe_trend_accepts_neutral_daily_pullback_with_bullish_macro_regime(self) -> None:
        dates = pd.date_range("2020-01-01", periods=1_500, freq="B", tz="UTC")
        rising_closes = [100 * 1.003**index for index in range(1_440)]
        frame = pd.DataFrame({"close": [*rising_closes, *([rising_closes[-1]] * 60)]}, index=dates)

        trend = _multi_timeframe_trend_status(frame, 20, 50, 5, 0.01)

        self.assertEqual(trend, ("bullish", trend[1], "neutral_pullback"))

    def test_interrupted_trend_qualification_retains_previous_qualified_members(self) -> None:
        core = Instrument(symbol="CORE", label="CORE", asset_name="Core")
        previous = Instrument(
            symbol="PREVIOUS",
            label="PREVIOUS",
            asset_name="Previous",
            discovery_source="watchlist_candidate",
            trend_direction="bullish",
        )
        candidate = Instrument(symbol="CANDIDATE", label="CANDIDATE", asset_name="Candidate")
        with TemporaryDirectory() as directory:
            database_path = Path(directory) / "universe.db"
            UniverseStore(database_path).replace([core, previous])
            config = SimpleNamespace(
                watchlist=[core, candidate],
                universe=SimpleNamespace(
                    enabled=True,
                    core_symbols=["CORE"],
                    watchlist_candidates=["CANDIDATE"],
                    watchlist_candidate_limit=1,
                    database_path=str(database_path),
                    max_dynamic_candidates=20,
                    sources=[{"name": "most_active", "enabled": True}],
                    trend_filter={"enabled": True, "max_candidates": 1, "minimum_qualification_coverage_pct": 0.8},
                ),
            )
            provider = SimpleNamespace(
                scan_instruments=lambda _: [],
                fetch_ohlcv=lambda *_: (_ for _ in ()).throw(RuntimeError("Gateway disconnect")),
            )

            result = refresh_universe(config, provider)

        self.assertTrue(result.retained_previous_snapshot)
        self.assertEqual([instrument.symbol for instrument in result.instruments], ["CORE", "PREVIOUS"])

    def test_watchlist_candidates_are_discovery_members_that_remain_trend_qualified(self) -> None:
        core = Instrument(symbol="CORE", label="CORE", asset_name="Core")
        candidate = Instrument(symbol="CANDIDATE", label="CANDIDATE", asset_name="Candidate")
        config = SimpleNamespace(
            watchlist=[core, candidate],
            universe=SimpleNamespace(watchlist_candidates=["CANDIDATE"]),
        )

        candidates = _watchlist_candidates(config, {"|||"})

        self.assertEqual([instrument.symbol for instrument in candidates], ["CANDIDATE"])
        self.assertEqual(candidates[0].discovery_source, "watchlist_candidate")
        self.assertEqual(candidates[0].discovery_priority, 5)


if __name__ == "__main__":
    unittest.main()