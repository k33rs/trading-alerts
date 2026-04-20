from __future__ import annotations

from pathlib import Path
import json
from datetime import datetime, timezone


class AlertState:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.seen, self.fetch_times, self.fetch_failure_times, self.hot_trend_scores, self.near_setup_scores = self._load()

    def _load(self) -> tuple[set[str], dict[str, str], dict[str, float], dict[str, float], dict[str, float]]:
        if not self.path.exists():
            return set(), {}, {}, {}, {}
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return (
            set(payload.get("seen", [])),
            dict(payload.get("fetch_times", {})),
            dict(payload.get("fetch_failure_times", {})),
            {symbol: float(score) for symbol, score in payload.get("hot_trend_scores", {}).items()},
            {key: float(score) for key, score in payload.get("near_setup_scores", {}).items()},
        )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(
                {
                    "seen": sorted(self.seen),
                    "fetch_times": self.fetch_times,
                    "fetch_failure_times": self.fetch_failure_times,
                    "hot_trend_scores": self.hot_trend_scores,
                    "near_setup_scores": self.near_setup_scores,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.path)

    def has_seen(self, key: str) -> bool:
        return key in self.seen

    def remember(self, key: str) -> None:
        self.seen.add(key)
        self._save()

    def last_fetched_at(self, key: str) -> datetime | None:
        return self._timestamp(self.fetch_times.get(key))

    def last_failed_at(self, key: str) -> datetime | None:
        return self._timestamp(self.fetch_failure_times.get(key))

    def mark_fetched(self, key: str, fetched_at: datetime) -> None:
        self.fetch_times[key] = self._timestamp_value(fetched_at)
        self.fetch_failure_times.pop(key, None)
        self._save()

    def mark_fetch_failed(self, key: str, failed_at: datetime) -> None:
        self.fetch_failure_times[key] = self._timestamp_value(failed_at)
        self._save()

    def hot_trend_score(self, symbol: str) -> float:
        return self.hot_trend_scores.get(symbol, 0.0)

    def mark_hot_trend_score(self, symbol: str, score: float) -> None:
        self.hot_trend_scores[symbol] = round(float(score), 4)
        self._save()

    def near_setup_score(self, key: str) -> float:
        return self.near_setup_scores.get(key, 0.0)

    def mark_near_setup_score(self, key: str, score: float) -> None:
        self.near_setup_scores[key] = round(float(score), 4)
        self._save()

    @staticmethod
    def _timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    @staticmethod
    def _timestamp_value(timestamp: datetime) -> str:
        resolved = timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=timezone.utc)
        return resolved.isoformat()