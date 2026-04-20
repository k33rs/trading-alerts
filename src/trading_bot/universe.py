from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

import pandas as pd

from trading_bot.config import BotConfig
from trading_bot.data_provider import MarketDataRouter
from trading_bot.models import Instrument


@dataclass(slots=True)
class UniverseRefreshResult:
    instruments: list[Instrument]
    source_counts: dict[str, int]
    failures: list[str]
    retained_previous_snapshot: bool = False


class UniverseStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def replace(self, instruments: list[Instrument]) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS universe_members (
                    member_key TEXT PRIMARY KEY,
                    instrument_json TEXT NOT NULL,
                    refreshed_at TEXT NOT NULL
                )
                """
            )
            refreshed_at = datetime.now(timezone.utc).isoformat()
            connection.execute("DELETE FROM universe_members")
            connection.executemany(
                "INSERT INTO universe_members(member_key, instrument_json, refreshed_at) VALUES (?, ?, ?)",
                [(_instrument_key(instrument), json.dumps(asdict(instrument), sort_keys=True), refreshed_at) for instrument in instruments],
            )

    def write_json_snapshot(self, instruments: list[Instrument]) -> None:
        snapshot_path = self.database_path.with_suffix(".json")
        temporary_path = snapshot_path.with_suffix(".json.tmp")
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps([universe_instrument_payload(instrument) for instrument in instruments], indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(snapshot_path)

    def load(self) -> list[Instrument]:
        if not self.database_path.exists():
            return []
        with sqlite3.connect(self.database_path) as connection:
            try:
                rows = connection.execute(
                    "SELECT instrument_json FROM universe_members ORDER BY member_key"
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return _order_instruments([Instrument(**json.loads(row[0])) for row in rows])

    def load_fresh(self, max_age_seconds: int, now: datetime | None = None) -> list[Instrument]:
        if not self.database_path.exists():
            return []
        with sqlite3.connect(self.database_path) as connection:
            try:
                rows = connection.execute(
                    "SELECT instrument_json, refreshed_at FROM universe_members ORDER BY member_key"
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        if not rows:
            return []
        refreshed_at = datetime.fromisoformat(rows[0][1])
        current_time = now or datetime.now(timezone.utc)
        if refreshed_at.tzinfo is None:
            refreshed_at = refreshed_at.replace(tzinfo=timezone.utc)
        if current_time - refreshed_at > timedelta(seconds=max_age_seconds):
            return []
        return _order_instruments([Instrument(**json.loads(row[0])) for row in rows])


def refresh_universe(config: BotConfig, provider: MarketDataRouter) -> UniverseRefreshResult:
    core = _core_instruments(config)
    members = list(core)
    source_counts: dict[str, int] = {"core": len(core)}
    failures: list[str] = []
    attempted_sources = 0
    successful_sources = 0
    dynamic_limit = max(0, config.universe.max_dynamic_candidates)
    seen = {_instrument_key(instrument) for instrument in members}

    watchlist_candidates = _watchlist_candidates(config, seen)
    members.extend(watchlist_candidates)
    seen.update(_instrument_key(instrument) for instrument in watchlist_candidates)
    source_counts["watchlist_candidates"] = len(watchlist_candidates)

    for source in config.universe.sources:
        if not bool(source.get("enabled", False)) or len(members) - len(core) >= dynamic_limit:
            continue
        attempted_sources += 1
        source_name = str(source.get("name", source.get("scan_code", "unnamed")))
        try:
            discovered = provider.scan_instruments(source)
        except Exception as exc:
            failures.append(f"{source_name}: {exc}")
            continue
        successful_sources += 1
        added = 0
        for instrument in discovered:
            if len(members) - len(core) >= dynamic_limit:
                break
            key = _instrument_key(instrument)
            if key in seen:
                existing_index = next((index for index, member in enumerate(members) if _instrument_key(member) == key), None)
                if existing_index is not None and _discovery_order(instrument) < _discovery_order(members[existing_index]):
                    members[existing_index] = replace(instrument)
                continue
            seen.add(key)
            members.append(instrument)
            added += 1
        source_counts[source_name] = added

    trend_filter = getattr(config.universe, "trend_filter", {})
    trend_failures: list[str] = []
    evaluated_count = 0
    if bool(trend_filter.get("enabled", False)):
        dynamic_members = members[len(core):]
        qualified_members, rejected_count, trend_failures, evaluated_count = _filter_trending_instruments(
            dynamic_members,
            provider,
            trend_filter,
        )
        members = [*core, *qualified_members]
        source_counts["trend_rejected"] = rejected_count
        failures.extend(trend_failures)

    store = UniverseStore(Path(config.universe.database_path))
    minimum_coverage = min(1.0, max(0.0, float(trend_filter.get("minimum_qualification_coverage_pct", 0.8))))
    if evaluated_count and len(trend_failures) / evaluated_count > 1.0 - minimum_coverage:
        retained_members = _retained_qualified_members(store.load())
        if retained_members:
            source_counts["retained_snapshot"] = len(retained_members)
            store.replace(retained_members)
            store.write_json_snapshot(retained_members)
            failures.append(
                f"trend qualification coverage was insufficient: {evaluated_count - len(trend_failures)}/{evaluated_count} candidates completed"
            )
            return UniverseRefreshResult(
                instruments=retained_members,
                source_counts=source_counts,
                failures=failures,
                retained_previous_snapshot=True,
            )
    if attempted_sources and successful_sources == 0:
        previous_members = store.load()
        if previous_members:
            source_counts["retained_snapshot"] = len(previous_members)
            store.write_json_snapshot(previous_members)
            return UniverseRefreshResult(
                instruments=previous_members,
                source_counts=source_counts,
                failures=failures,
                retained_previous_snapshot=True,
            )
    members = _order_instruments(members)
    store.replace(members)
    store.write_json_snapshot(members)
    return UniverseRefreshResult(instruments=members, source_counts=source_counts, failures=failures)


def active_universe(config: BotConfig) -> list[Instrument]:
    if not config.universe.enabled:
        return list(config.watchlist)
    snapshot = UniverseStore(Path(config.universe.database_path)).load_fresh(config.universe.snapshot_max_age_seconds)
    return snapshot or _core_instruments(config)


def _core_instruments(config: BotConfig) -> list[Instrument]:
    if not config.universe.enabled or not config.universe.core_symbols:
        return list(config.watchlist)
    catalog = {instrument.symbol: instrument for instrument in config.watchlist}
    return [catalog[symbol] for symbol in config.universe.core_symbols]


def _watchlist_candidates(config: BotConfig, seen: set[str]) -> list[Instrument]:
    catalog = {instrument.symbol: instrument for instrument in config.watchlist}
    candidates: list[Instrument] = []
    candidate_limit = max(0, int(getattr(config.universe, "watchlist_candidate_limit", 0)))
    configured_candidates = getattr(config.universe, "watchlist_candidates", [])
    for rank, symbol in enumerate(configured_candidates[:candidate_limit or None]):
        instrument = catalog.get(symbol)
        if instrument is None or _instrument_key(instrument) in seen:
            continue
        candidates.append(
            replace(
                instrument,
                discovery_source="watchlist_candidate",
                discovery_priority=5,
                discovery_rank=rank,
            )
        )
    return candidates


def _retained_qualified_members(instruments: list[Instrument]) -> list[Instrument]:
    return [
        instrument
        for instrument in instruments
        if instrument.discovery_source is None or instrument.trend_direction is not None
    ]


def _instrument_key(instrument: Instrument) -> str:
    return "|".join(
        [
            instrument.provider or "",
            instrument.provider_symbol or instrument.symbol,
            instrument.asset_type or "",
            instrument.exchange or "",
            instrument.primary_exchange or "",
            instrument.currency or "",
        ]
    )


def universe_instrument_payload(instrument: Instrument) -> dict[str, object]:
    return {
        "symbol": instrument.symbol,
        "asset_type": instrument.asset_type,
        "exchange": instrument.exchange,
        "currency": instrument.currency,
        "source": instrument.discovery_source or "core",
        "source_priority": instrument.discovery_priority,
        "scanner_rank": instrument.discovery_rank,
        "trend_direction": instrument.trend_direction,
        "trend_strength": instrument.trend_strength,
        "daily_trend_context": instrument.daily_trend_context,
        "name": instrument.asset_name,
    }


def _order_instruments(instruments: list[Instrument]) -> list[Instrument]:
    return sorted(instruments, key=lambda instrument: (_discovery_order(instrument), instrument.symbol))


def _discovery_order(instrument: Instrument) -> tuple[int, int]:
    return (int(instrument.discovery_priority), int(instrument.discovery_rank))


def _filter_trending_instruments(
    instruments: list[Instrument],
    provider: MarketDataRouter,
    settings: dict,
) -> tuple[list[Instrument], int, list[str], int]:
    interval = str(settings.get("interval", "1d"))
    max_candidates = max(0, int(settings.get("max_candidates", len(instruments))))
    fast_length = max(2, int(settings.get("fast_ema", 20)))
    slow_length = max(fast_length + 1, int(settings.get("slow_ema", 50)))
    slope_lookback = max(1, int(settings.get("slope_lookback", 5)))
    min_separation_pct = max(0.0, float(settings.get("min_ema_separation_pct", 0.01)))
    qualified: list[Instrument] = []
    failures: list[str] = []
    evaluated = 0

    for instrument in _order_instruments(instruments):
        if evaluated >= max_candidates:
            break
        evaluated += 1
        try:
            frame = provider.fetch_ohlcv(instrument, interval)
        except Exception as exc:
            failures.append(f"trend/{instrument.symbol}: {exc}")
            continue
        trend = _multi_timeframe_trend_status(
            frame,
            fast_length,
            slow_length,
            slope_lookback,
            min_separation_pct,
        )
        if trend is None:
            continue
        direction, strength, daily_context = trend
        qualified.append(
            replace(
                instrument,
                trend_direction=direction,
                trend_strength=strength,
                daily_trend_context=daily_context,
            )
        )

    return qualified, evaluated - len(qualified), failures, evaluated


def _trend_status(
    frame: pd.DataFrame,
    fast_length: int,
    slow_length: int,
    slope_lookback: int,
    min_separation_pct: float,
) -> tuple[str, float] | None:
    if "close" not in frame or len(frame) < slow_length + slope_lookback:
        return None
    closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if len(closes) < slow_length + slope_lookback:
        return None
    fast_ema = closes.ewm(span=fast_length, adjust=False).mean()
    slow_ema = closes.ewm(span=slow_length, adjust=False).mean()
    latest_fast = float(fast_ema.iloc[-1])
    latest_slow = float(slow_ema.iloc[-1])
    if latest_slow <= 0:
        return None
    separation = abs(latest_fast - latest_slow) / latest_slow
    if separation < min_separation_pct:
        return None
    fast_rising = latest_fast > float(fast_ema.iloc[-1 - slope_lookback])
    slow_rising = latest_slow > float(slow_ema.iloc[-1 - slope_lookback])
    fast_falling = latest_fast < float(fast_ema.iloc[-1 - slope_lookback])
    slow_falling = latest_slow < float(slow_ema.iloc[-1 - slope_lookback])
    if latest_fast > latest_slow and fast_rising and slow_rising:
        return "bullish", separation
    if latest_fast < latest_slow and fast_falling and slow_falling:
        return "bearish", separation
    return None


def _multi_timeframe_trend_status(
    frame: pd.DataFrame,
    fast_length: int,
    slow_length: int,
    slope_lookback: int,
    min_separation_pct: float,
) -> tuple[str, float, str] | None:
    daily_trend = _trend_status(frame, fast_length, slow_length, slope_lookback, min_separation_pct)
    weekly_trend = _trend_status(
        _resample_closes(frame, "W-FRI"),
        fast_length,
        slow_length,
        slope_lookback,
        min_separation_pct,
    )
    monthly_trend = _trend_status(
        _resample_closes(frame, "ME"),
        fast_length,
        slow_length,
        slope_lookback,
        min_separation_pct,
    )
    if weekly_trend is None or monthly_trend is None:
        return None
    weekly_direction, weekly_strength = weekly_trend
    monthly_direction, monthly_strength = monthly_trend
    if weekly_direction != monthly_direction:
        return None
    daily_context = "neutral_pullback" if daily_trend is None else "aligned" if daily_trend[0] == weekly_direction else "pullback"
    return weekly_direction, min(weekly_strength, monthly_strength), daily_context


def _resample_closes(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if "close" not in frame or not isinstance(frame.index, pd.DatetimeIndex):
        return pd.DataFrame()
    return frame[["close"]].resample(frequency).last().dropna()