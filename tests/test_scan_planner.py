from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest

from trading_bot.main import _fetch_key, _fetch_plan, _next_scan_delay_seconds
from trading_bot.models import Instrument


class ScanPlannerTests(unittest.TestCase):
    def test_limits_requests_and_rotates_oldest_dynamic_pairs(self) -> None:
        instruments = [
            Instrument(symbol="CORE", label="CORE", asset_name="Core"),
            Instrument(symbol="FIRST", label="FIRST", asset_name="First"),
            Instrument(symbol="SECOND", label="SECOND", asset_name="Second"),
        ]
        state = SimpleNamespace(
            last_fetched_at=lambda key: {
                _fetch_key("CORE", "1h"): datetime(2026, 7, 17, 9, tzinfo=timezone.utc),
                _fetch_key("FIRST", "1h"): datetime(2026, 7, 17, 8, tzinfo=timezone.utc),
                _fetch_key("SECOND", "1h"): datetime(2026, 7, 17, 7, tzinfo=timezone.utc),
            }.get(key),
            last_failed_at=lambda _: None,
        )
        config = SimpleNamespace(
            app=SimpleNamespace(historical_request_budget=2, timeframe_refresh_seconds={"1h": 3600}),
            universe=SimpleNamespace(enabled=True, core_symbols=["CORE"]),
        )

        plan = _fetch_plan(config, state, instruments, ["1h"], datetime(2026, 7, 17, 10, tzinfo=timezone.utc))

        self.assertEqual([(instrument.symbol, timeframe) for instrument, timeframe in plan], [("CORE", "1h"), ("SECOND", "1h")])

    def test_skips_pairs_until_their_timeframe_cadence_is_due(self) -> None:
        instrument = Instrument(symbol="TEST", label="TEST", asset_name="Test")
        now = datetime(2026, 7, 17, 10, tzinfo=timezone.utc)
        state = SimpleNamespace(last_fetched_at=lambda _: now - timedelta(minutes=30), last_failed_at=lambda _: None)
        config = SimpleNamespace(
            app=SimpleNamespace(historical_request_budget=20, timeframe_refresh_seconds={"1h": 3600}),
            universe=SimpleNamespace(enabled=False, core_symbols=[]),
        )

        plan = _fetch_plan(config, state, [instrument], ["1h"], now)

        self.assertEqual(plan, [])

    def test_prioritizes_shorter_timeframes_for_new_dynamic_candidates(self) -> None:
        instruments = [
            Instrument(symbol="FIRST", label="FIRST", asset_name="First"),
            Instrument(symbol="SECOND", label="SECOND", asset_name="Second"),
        ]
        state = SimpleNamespace(last_fetched_at=lambda _: None, last_failed_at=lambda _: None)
        config = SimpleNamespace(
            app=SimpleNamespace(
                historical_request_budget=2,
                timeframe_refresh_seconds={"1h": 3600, "1d": 86400},
            ),
            universe=SimpleNamespace(enabled=True, core_symbols=[]),
        )

        plan = _fetch_plan(
            config,
            state,
            instruments,
            ["1h", "1d"],
            datetime(2026, 7, 17, 10, tzinfo=timezone.utc),
        )

        self.assertEqual([(instrument.symbol, timeframe) for instrument, timeframe in plan], [("FIRST", "1h"), ("SECOND", "1h")])

    def test_defers_recent_failures_without_blocking_new_pairs(self) -> None:
        now = datetime(2026, 7, 17, 10, tzinfo=timezone.utc)
        instruments = [
            Instrument(symbol="FAILED", label="FAILED", asset_name="Failed"),
            Instrument(symbol="NEW", label="NEW", asset_name="New"),
        ]
        state = SimpleNamespace(
            last_fetched_at=lambda _: None,
            last_failed_at=lambda key: now - timedelta(minutes=30) if key == _fetch_key("FAILED", "1h") else None,
        )
        config = SimpleNamespace(
            app=SimpleNamespace(
                historical_request_budget=2,
                historical_failure_retry_seconds=3600,
                timeframe_refresh_seconds={"1h": 3600},
            ),
            universe=SimpleNamespace(enabled=True, core_symbols=[]),
        )

        plan = _fetch_plan(config, state, instruments, ["1h"], now)

        self.assertEqual([(instrument.symbol, timeframe) for instrument, timeframe in plan], [("NEW", "1h")])

    def test_prioritizes_new_candidates_from_hot_discovery_sources(self) -> None:
        instruments = [
            Instrument(symbol="QUIET", label="QUIET", asset_name="Quiet", discovery_priority=30, discovery_rank=0),
            Instrument(symbol="HOT", label="HOT", asset_name="Hot", discovery_priority=10, discovery_rank=4),
        ]
        state = SimpleNamespace(last_fetched_at=lambda _: None, last_failed_at=lambda _: None)
        config = SimpleNamespace(
            app=SimpleNamespace(historical_request_budget=1, timeframe_refresh_seconds={"1h": 3600}),
            universe=SimpleNamespace(enabled=True, core_symbols=[]),
        )

        plan = _fetch_plan(config, state, instruments, ["1h"], datetime(2026, 7, 17, 10, tzinfo=timezone.utc))

        self.assertEqual([(instrument.symbol, timeframe) for instrument, timeframe in plan], [("HOT", "1h")])

    def test_prioritizes_confirmed_strong_trends_with_equal_staleness(self) -> None:
        now = datetime(2026, 7, 17, 10, tzinfo=timezone.utc)
        instruments = [
            Instrument(symbol="QUIET", label="QUIET", asset_name="Quiet"),
            Instrument(symbol="TRENDING", label="TRENDING", asset_name="Trending"),
        ]
        state = SimpleNamespace(
            last_fetched_at=lambda _: now - timedelta(hours=2),
            last_failed_at=lambda _: None,
            hot_trend_score=lambda symbol: 85.0 if symbol == "TRENDING" else 10.0,
        )
        config = SimpleNamespace(
            app=SimpleNamespace(historical_request_budget=1, timeframe_refresh_seconds={"1h": 3600}),
            universe=SimpleNamespace(enabled=True, core_symbols=[]),
        )

        plan = _fetch_plan(config, state, instruments, ["1h"], now)

        self.assertEqual([(instrument.symbol, timeframe) for instrument, timeframe in plan], [("TRENDING", "1h")])

    def test_prioritizes_candidates_near_an_enabled_pattern_setup(self) -> None:
        now = datetime(2026, 7, 17, 10, tzinfo=timezone.utc)
        instruments = [
            Instrument(symbol="QUIET", label="QUIET", asset_name="Quiet"),
            Instrument(symbol="NEAR", label="NEAR", asset_name="Near"),
        ]
        state = SimpleNamespace(
            last_fetched_at=lambda _: now - timedelta(hours=2),
            last_failed_at=lambda _: None,
            near_setup_score=lambda key: 90.0 if key == _fetch_key("NEAR", "1h") else 20.0,
            hot_trend_score=lambda _: 0.0,
        )
        config = SimpleNamespace(
            app=SimpleNamespace(historical_request_budget=1, timeframe_refresh_seconds={"1h": 3600}),
            universe=SimpleNamespace(enabled=True, core_symbols=[]),
        )

        plan = _fetch_plan(config, state, instruments, ["1h"], now)

        self.assertEqual([(instrument.symbol, timeframe) for instrument, timeframe in plan], [("NEAR", "1h")])

    def test_waits_until_the_next_due_fetch_but_not_longer_than_idle_cap(self) -> None:
        now = datetime(2026, 7, 17, 10, tzinfo=timezone.utc)
        instrument = Instrument(symbol="TEST", label="TEST", asset_name="Test")
        state = SimpleNamespace(
            last_fetched_at=lambda _: now - timedelta(minutes=30),
            last_failed_at=lambda _: None,
        )
        config = SimpleNamespace(
            app=SimpleNamespace(scan_interval_seconds=300, timeframe_refresh_seconds={"1h": 3600}),
        )

        delay = _next_scan_delay_seconds(config, state, [instrument], ["1h"], now)

        self.assertEqual(delay, 300.0)

    def test_runs_immediately_while_other_due_pairs_remain(self) -> None:
        now = datetime(2026, 7, 17, 10, tzinfo=timezone.utc)
        instrument = Instrument(symbol="TEST", label="TEST", asset_name="Test")
        state = SimpleNamespace(last_fetched_at=lambda _: None, last_failed_at=lambda _: None)
        config = SimpleNamespace(
            app=SimpleNamespace(scan_interval_seconds=300, timeframe_refresh_seconds={"1h": 3600}),
        )

        delay = _next_scan_delay_seconds(config, state, [instrument], ["1h"], now)

        self.assertEqual(delay, 0.0)


if __name__ == "__main__":
    unittest.main()