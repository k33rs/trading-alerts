from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from trading_bot.config import load_config
from trading_bot.universe import UniverseRefreshResult
from trading_bot import main as scanner
from trading_bot import universe_service


class ServiceRecoveryTests(unittest.TestCase):
    def test_universe_clears_failed_connection_and_retries_early(self) -> None:
        config = SimpleNamespace(universe=SimpleNamespace(enabled=True, client_id=18), data_source=SimpleNamespace())
        args = SimpleNamespace(config="unused", list=False, account_mode="paper", once=False, interval_seconds=14400, retry_interval_seconds=300)
        provider = Mock()
        failed = UniverseRefreshResult([], {}, ["Gateway disconnected"], True)
        successful = UniverseRefreshResult([], {}, [])
        with patch.object(universe_service, "configure_logging"), \
             patch.object(universe_service, "log"), \
             patch.object(universe_service, "parse_args", return_value=args), \
             patch.object(universe_service, "load_config", return_value=config), \
             patch.object(universe_service, "MarketDataRouter", return_value=provider), \
             patch.object(universe_service, "refresh_universe", side_effect=[failed, successful]) as refresh, \
             patch.object(universe_service.time, "sleep", side_effect=[None, StopIteration]) as sleep:
            with self.assertRaises(StopIteration):
                universe_service.main()
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [300, 14400])
        self.assertEqual(refresh.call_count, 2)
        self.assertEqual(provider.close.call_count, 2)

    def test_scanner_reconnects_after_interrupted_scan(self) -> None:
        config = load_config(Path(__file__).resolve().parents[1] / "config" / "live-data.yaml")
        args = SimpleNamespace(config="unused", state_file="unused", account_mode="paper", once=False, diagnostics=False)
        provider = Mock()
        with patch.object(scanner, "configure_logging"), \
             patch.object(scanner, "log"), \
             patch.object(scanner, "parse_args", return_value=args), \
             patch.object(scanner, "load_config", return_value=config), \
             patch.object(scanner, "MarketDataRouter", return_value=provider), \
             patch.object(scanner, "AlertState"), \
             patch.object(scanner, "TelegramNotifier"), \
             patch.object(scanner, "AlertReviewStore"), \
             patch.object(scanner, "active_universe", return_value=[]), \
             patch.object(scanner, "run_scan", side_effect=[RuntimeError("Gateway disconnected"), []]) as scan, \
             patch.object(scanner.time, "sleep", side_effect=[None, StopIteration]) as sleep:
            with self.assertRaises(StopIteration):
                scanner.main()
        self.assertEqual(scan.call_count, 2)
        self.assertEqual(provider.close.call_count, 2)
        self.assertEqual(sleep.call_args_list[0].args[0], 300)

    def test_once_surfaces_failure_without_retrying(self) -> None:
        args = SimpleNamespace(config="unused", state_file="unused", account_mode="paper", once=True, diagnostics=False)
        provider = Mock()
        with patch.object(scanner, "configure_logging"), \
             patch.object(scanner, "parse_args", return_value=args), \
             patch.object(scanner, "load_config"), \
             patch.object(scanner, "MarketDataRouter", return_value=provider), \
             patch.object(scanner, "AlertState"), \
             patch.object(scanner, "TelegramNotifier"), \
             patch.object(scanner, "AlertReviewStore"), \
             patch.object(scanner, "run_scan", side_effect=RuntimeError("Gateway disconnected")), \
             patch.object(scanner.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "Gateway disconnected"):
                scanner.main()
        provider.close.assert_called_once()
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
