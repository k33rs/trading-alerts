from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from trading_bot.backtest import BacktestSummary, _settle_trade, load_ohlcv_csv, run_backtest
from trading_bot.models import Instrument, TradeSignal


class BacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        instrument = Instrument(symbol="TEST", label="TEST", asset_name="Test instrument")
        self.signal = TradeSignal(
            pattern="break_in",
            direction="long",
            timeframe="1d",
            instrument=instrument,
            candle_time="2026-01-01T00:00:00+00:00",
            entry=100.0,
            stop_loss=95.0,
            target=110.0,
            rr=2.0,
            position_size=None,
        )

    def test_csv_loader_normalizes_standard_ohlcv_columns(self) -> None:
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "bars.csv"
            csv_path.write_text(
                "Date,Open,High,Low,Close,Volume\n2026-01-01,100,101,99,100.5,1000\n",
                encoding="utf-8",
            )

            frame = load_ohlcv_csv(csv_path)

        self.assertEqual(list(frame.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(str(frame.index.tz), "UTC")

    def test_same_bar_stop_and_target_is_conservative_stop(self) -> None:
        future_bars = pd.DataFrame(
            {"open": [101.0], "high": [111.0], "low": [94.0], "close": [109.0], "volume": [1.0]},
            index=pd.DatetimeIndex(["2026-01-02T00:00:00Z"]),
        )

        trade = _settle_trade(self.signal, future_bars)

        self.assertEqual(trade.outcome, "stop")
        self.assertEqual(trade.realized_r, -1.0)

    def test_adverse_slippage_reduces_target_reward(self) -> None:
        future_bars = pd.DataFrame(
            {"open": [101.0], "high": [111.0], "low": [99.0], "close": [109.0], "volume": [1.0]},
            index=pd.DatetimeIndex(["2026-01-02T00:00:00Z"]),
        )

        ideal_trade = _settle_trade(self.signal, future_bars)
        slipped_trade = _settle_trade(self.signal, future_bars, slippage_bps=10.0)

        self.assertEqual(ideal_trade.realized_r, 2.0)
        self.assertLess(float(slipped_trade.realized_r), float(ideal_trade.realized_r))

    def test_summary_reports_completed_and_unresolved_trades(self) -> None:
        target_bars = pd.DataFrame(
            {"open": [101.0], "high": [111.0], "low": [99.0], "close": [109.0], "volume": [1.0]},
            index=pd.DatetimeIndex(["2026-01-02T00:00:00Z"]),
        )
        completed = _settle_trade(self.signal, target_bars)
        unresolved = _settle_trade(self.signal, target_bars.iloc[0:0])

        report = BacktestSummary("TEST", "1d", [completed, unresolved]).to_dict()

        self.assertEqual(report["completed_trade_count"], 1)
        self.assertEqual(report["wins"], 1)
        self.assertEqual(report["unresolved"], 1)
        self.assertEqual(report["total_r"], 2.0)

    def test_replay_settles_delayed_signal_from_its_own_next_bar(self) -> None:
        index = pd.date_range("2026-01-01", periods=4, freq="D", tz="UTC")
        frame = pd.DataFrame(
            {"open": [100.0, 100.0, 101.0, 101.0], "high": [101.0, 101.0, 111.0, 101.0], "low": [99.0, 99.0, 100.0, 100.0], "close": [100.0, 100.0, 110.0, 101.0], "volume": [1.0, 1.0, 1.0, 1.0]},
            index=index,
        )
        config = SimpleNamespace(
            patterns={"break_out": SimpleNamespace(name="break_out", enabled=True, settings={"timeframes": ["1d"]})},
            app=SimpleNamespace(risk_per_trade_pct=0.01, position_value_pct=0.01),
        )

        def delayed_signal(*args):
            available = args[1]
            if len(available) != 3:
                return []
            return [
                TradeSignal(
                    pattern="break_out",
                    direction="long",
                    timeframe="1d",
                    instrument=self.signal.instrument,
                    candle_time=index[1].isoformat(),
                    entry=100.0,
                    stop_loss=95.0,
                    target=110.0,
                    rr=2.0,
                    position_size=None,
                )
            ]

        with patch("trading_bot.backtest.run_pattern", side_effect=delayed_signal):
            summary = run_backtest(frame, config, self.signal.instrument, "1d", 100_000.0, warmup_bars=2)

        self.assertEqual(len(summary.trades), 1)
        self.assertEqual(summary.trades[0].outcome, "target")
        self.assertEqual(summary.trades[0].exit_time, index[2].isoformat())


if __name__ == "__main__":
    unittest.main()