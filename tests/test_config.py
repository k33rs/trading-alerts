from pathlib import Path
import unittest

from trading_bot.config import load_config
from trading_bot.universe import _core_instruments, _instrument_key, _watchlist_candidates


class LiveDataConfigTests(unittest.TestCase):
    def test_live_universe_covers_every_configured_candidate_and_crypto(self) -> None:
        config = load_config(Path(__file__).resolve().parents[1] / "config" / "live-data.yaml")
        core = _core_instruments(config)
        candidates = _watchlist_candidates(config, {_instrument_key(i) for i in core})
        symbols = {i.symbol for i in [*core, *candidates]}
        self.assertTrue(set(config.universe.watchlist_candidates) <= symbols)
        self.assertTrue({i.symbol for i in config.watchlist if i.asset_type == "crypto"} <= symbols)
        self.assertIn("MES", symbols)
        self.assertIn("DTE", symbols)

    def test_ibkr_defaults_to_regular_trading_hours(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config" / "live-data.yaml"

        config = load_config(config_path)

        self.assertTrue(config.data_source.providers["ibkr"]["default_use_rth"])
