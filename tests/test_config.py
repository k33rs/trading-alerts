from pathlib import Path
import unittest

from trading_bot.config import load_config


class LiveDataConfigTests(unittest.TestCase):
    def test_ibkr_defaults_to_regular_trading_hours(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "live-data.yaml"

        config = load_config(config_path)

        self.assertTrue(config.data_source.providers["ibkr"]["default_use_rth"])