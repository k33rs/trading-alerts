from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import pandas as pd

from trading_bot.models import TradeSignal


REVIEW_STATUSES = {"pending", "accepted", "rejected", "ambiguous"}
REVIEW_REPORT_DIMENSIONS = ("pattern", "timeframe", "asset_type", "discovery_source", "direction")


class AlertReviewStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def record(self, signal: TradeSignal, frame: pd.DataFrame) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS alert_reviews (
                    dedupe_key TEXT PRIMARY KEY,
                    recorded_at TEXT NOT NULL,
                    signal_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    review_status TEXT NOT NULL DEFAULT 'pending',
                    review_note TEXT NOT NULL DEFAULT '',
                    reviewed_at TEXT
                )
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO alert_reviews(
                    dedupe_key, recorded_at, signal_json, evidence_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    signal.dedupe_key,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(_signal_payload(signal), sort_keys=True, default=str),
                    json.dumps(_evidence_payload(signal, frame), sort_keys=True, default=str),
                ),
            )

    def list(self, status: str | None = None) -> list[dict[str, object]]:
        if status is not None and status not in REVIEW_STATUSES:
            raise ValueError(f"Unknown review status '{status}'.")
        if not self.database_path.exists():
            return []
        with sqlite3.connect(self.database_path) as connection:
            query = "SELECT dedupe_key, recorded_at, signal_json, evidence_json, review_status, review_note, reviewed_at FROM alert_reviews"
            parameters: tuple[str, ...] = ()
            if status:
                query += " WHERE review_status = ?"
                parameters = (status,)
            query += " ORDER BY recorded_at DESC"
            rows = connection.execute(query, parameters).fetchall()
        return [
            {
                "dedupe_key": row[0],
                "recorded_at": row[1],
                "signal": json.loads(row[2]),
                "evidence": json.loads(row[3]),
                "review_status": row[4],
                "review_note": row[5],
                "reviewed_at": row[6],
            }
            for row in rows
        ]

    def review(self, dedupe_key: str, status: str, note: str = "") -> None:
        if status not in REVIEW_STATUSES - {"pending"}:
            raise ValueError("Review status must be accepted, rejected, or ambiguous.")
        with sqlite3.connect(self.database_path) as connection:
            result = connection.execute(
                "UPDATE alert_reviews SET review_status = ?, review_note = ?, reviewed_at = ? WHERE dedupe_key = ?",
                (status, note, datetime.now(timezone.utc).isoformat(), dedupe_key),
            )
        if result.rowcount != 1:
            raise ValueError(f"No alert review record found for '{dedupe_key}'.")

    def report(self) -> dict[str, object]:
        return review_quality_report(self.list())


def review_quality_report(records: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, ...], dict[str, int]] = defaultdict(lambda: {status: 0 for status in REVIEW_STATUSES})
    for record in records:
        signal = record.get("signal", {})
        if not isinstance(signal, dict):
            continue
        status = str(record.get("review_status", "pending"))
        if status not in REVIEW_STATUSES:
            continue
        key = tuple(str(signal.get(dimension) or "unknown") for dimension in REVIEW_REPORT_DIMENSIONS)
        grouped[key][status] += 1

    groups: list[dict[str, object]] = []
    totals = {status: 0 for status in REVIEW_STATUSES}
    for key, counts in sorted(grouped.items()):
        assessed = counts["accepted"] + counts["rejected"]
        for status in REVIEW_STATUSES:
            totals[status] += counts[status]
        groups.append(
            {
                **dict(zip(REVIEW_REPORT_DIMENSIONS, key, strict=True)),
                "alerts": sum(counts.values()),
                **counts,
                "assessed": assessed,
                "acceptance_rate": counts["accepted"] / assessed if assessed else None,
            }
        )

    assessed_total = totals["accepted"] + totals["rejected"]
    return {
        "alerts": sum(totals.values()),
        **totals,
        "assessed": assessed_total,
        "acceptance_rate": totals["accepted"] / assessed_total if assessed_total else None,
        "groups": groups,
    }


def _signal_payload(signal: TradeSignal) -> dict[str, object]:
    return {
        "symbol": signal.instrument.symbol,
        "label": signal.instrument.label,
        "asset_name": signal.instrument.asset_name,
        "asset_type": signal.instrument.asset_type,
        "exchange": signal.instrument.exchange,
        "discovery_source": signal.instrument.discovery_source,
        "timeframe": signal.timeframe,
        "pattern": signal.pattern,
        "direction": signal.direction,
        "candle_time": signal.candle_time,
        "entry": signal.entry,
        "stop": signal.stop_loss,
        "target": signal.target,
        "rr": signal.rr,
        "notes": signal.notes,
        "metadata": signal.metadata,
    }


def _evidence_payload(signal: TradeSignal, frame: pd.DataFrame) -> dict[str, object]:
    candle_time = pd.Timestamp(signal.candle_time)
    index = frame.index
    if candle_time.tzinfo is None and getattr(index, "tz", None) is not None:
        candle_time = candle_time.tz_localize(index.tz)
    if candle_time.tzinfo is not None and getattr(index, "tz", None) is None:
        candle_time = candle_time.tz_convert(None)
    position = index.get_indexer([candle_time])[0]
    if position < 0:
        position = len(frame) - 1
    candle = frame.iloc[position]
    candle_range = float(candle["high"] - candle["low"])
    average_volume = frame["volume"].iloc[max(0, position - 20) : position].mean()
    volume_ratio = float(candle["volume"] / average_volume) if pd.notna(average_volume) and average_volume > 0 else None
    return {
        "candle": {
            "time": pd.Timestamp(candle.name).isoformat(),
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "volume": float(candle["volume"]),
            "body_to_range": abs(float(candle["close"] - candle["open"])) / candle_range if candle_range > 0 else 0.0,
            "volume_ratio_20": volume_ratio,
        },
        "trendlines": signal.metadata.get("chart_trendlines", []),
        "levels": signal.metadata.get("chart_levels", []),
        "fibonacci": signal.metadata.get("fibonacci"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review and classify persisted trading alerts")
    parser.add_argument("--database", default="data/alert-reviews.db", help="Path to the alert-review SQLite database")
    parser.add_argument("--list", action="store_true", help="Print review records as JSON")
    parser.add_argument("--report", action="store_true", help="Print grouped manual-review quality metrics as JSON")
    parser.add_argument("--status", choices=sorted(REVIEW_STATUSES), help="Filter listed records by review status")
    parser.add_argument("--record", help="Alert dedupe key to classify")
    parser.add_argument("--outcome", choices=sorted(REVIEW_STATUSES - {"pending"}), help="Manual review outcome")
    parser.add_argument("--note", default="", help="Reviewer note saved with --record")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = AlertReviewStore(Path(args.database))
    if args.record:
        if not args.outcome:
            raise ValueError("--outcome is required with --record.")
        store.review(args.record, args.outcome, args.note)
        return
    if args.outcome:
        raise ValueError("--outcome requires --record.")
    if args.report:
        if args.status:
            raise ValueError("--status cannot be combined with --report.")
        print(json.dumps(store.report(), indent=2))
        return
    print(json.dumps(store.list(args.status), indent=2))


if __name__ == "__main__":
    main()