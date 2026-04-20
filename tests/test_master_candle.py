import unittest

import pandas as pd

from trading_bot.patterns import _master_candle_status


SETTINGS = {
    "master_candle_min_body_to_range": 0.15,
    "master_candle_strong_body_to_range": 0.55,
    "master_candle_close_zone": 0.25,
    "master_candle_uncertainty_body_max": 0.12,
    "master_candle_spinning_top_body_max": 0.25,
    "master_candle_uncertainty_wick_min": 0.30,
}


class MasterCandleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = pd.Series({"open": 12.10, "high": 12.25, "low": 11.70, "close": 11.80})

    def test_green_long_legged_doji_cannot_confirm_short_break_in(self) -> None:
        candle = pd.Series({"open": 11.68, "high": 12.32, "low": 11.12, "close": 11.72})

        status = _master_candle_status(candle, self.previous, "short", SETTINGS)

        self.assertFalse(status["passed"])
        self.assertEqual(status["uncertainty_patterns"], "doji, long_legged_doji")

    def test_bearish_lower_zone_master_can_confirm_short_break_in(self) -> None:
        candle = pd.Series({"open": 12.10, "high": 12.20, "low": 11.10, "close": 11.25})

        status = _master_candle_status(candle, self.previous, "short", SETTINGS)

        self.assertTrue(status["passed"])


if __name__ == "__main__":
    unittest.main()