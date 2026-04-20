from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from trading_bot.state import AlertState


class AlertStateTests(unittest.TestCase):
    def test_writes_complete_json_snapshot_atomically(self) -> None:
        with TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            state = AlertState(state_path)
            state.mark_fetched("TEST|1h", datetime(2026, 8, 26, tzinfo=timezone.utc))

            payload = json.loads(state_path.read_text(encoding="utf-8"))

            self.assertEqual(payload["fetch_times"]["TEST|1h"], "2026-08-26T00:00:00+00:00")
            self.assertFalse(state_path.with_suffix(".json.tmp").exists())