from datetime import datetime, timezone
import unittest

import pandas as pd

from trading_bot.data_provider import confirmed_ohlcv_frame
from trading_bot.charts import _master_candle_timestamp
from trading_bot.models import Instrument, TradeSignal
from trading_bot.patterns import _bollinger_lower_band_pierced, _minimum_reward_risk


class CandleConfirmationTests(unittest.TestCase):
    def test_removes_latest_unclosed_bar_even_when_ohlc_values_exist(self) -> None:
        index = pd.to_datetime(["2026-07-20T08:00:00Z", "2026-07-20T12:00:00Z"])
        frame = pd.DataFrame(
            {
                "open": [10.0, 10.5],
                "high": [10.8, 11.2],
                "low": [9.8, 10.3],
                "close": [10.6, 11.0],
                "volume": [100.0, 150.0],
            },
            index=index,
        )

        confirmed = confirmed_ohlcv_frame(frame, "4h", datetime(2026, 7, 20, 14, tzinfo=timezone.utc))

        self.assertEqual(list(confirmed.index), [index[0]])

    def test_chart_uses_recorded_master_candle_timestamp(self) -> None:
        signal = TradeSignal(
            pattern="break_out",
            direction="long",
            timeframe="4h",
            instrument=Instrument(symbol="TEST", label="TEST", asset_name="Test"),
            candle_time="2026-07-20T16:00:00+00:00",
            entry=10.0,
            stop_loss=9.0,
            target=12.0,
            rr=2.0,
            position_size=None,
            metadata={"master_candle": {"time": "2026-07-20T12:00:00+00:00", "open": 9.5, "close": 10.0}},
        )

        self.assertEqual(_master_candle_timestamp(signal), pd.Timestamp("2026-07-20T12:00:00+00:00"))

    def test_bollinger_requires_an_actual_lower_band_pierce(self) -> None:
        near_band = pd.Series({"low": 100.01, "lower": 100.0})
        pierced_band = pd.Series({"low": 100.0, "lower": 100.0})

        self.assertFalse(_bollinger_lower_band_pierced(near_band))
        self.assertTrue(_bollinger_lower_band_pierced(pierced_band))

    def test_bollinger_minimum_reward_risk_matches_the_sizing_model(self) -> None:
        self.assertEqual(_minimum_reward_risk({}), 1.5)
        self.assertEqual(_minimum_reward_risk({"minimum_rr": 2.0}), 2.0)