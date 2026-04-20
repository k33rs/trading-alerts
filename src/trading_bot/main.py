from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
import time

import pandas as pd

from trading_bot.charts import render_signal_chart
from trading_bot.config import BotConfig, load_config
from trading_bot.data_provider import MarketDataRouter
from trading_bot.models import TradeSignal
from trading_bot.patterns import diagnose_pattern, run_pattern
from trading_bot.review import AlertReviewStore
from trading_bot.state import AlertState
from trading_bot.telegram_client import TelegramNotifier
from trading_bot.universe import active_universe


LOG_FILE_PATH = Path("data/logs/trading-bot.log")
DEFAULT_TIMEFRAME_REFRESH_SECONDS = {
    "1h": 3_600,
    "2h": 7_200,
    "4h": 14_400,
    "1d": 86_400,
    "1wk": 604_800,
    "1mo": 2_592_000,
}


def configure_logging() -> None:
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler],
        force=True,
    )


def log(message: str) -> None:
    logger = logging.getLogger("trading_bot")
    for line in message.splitlines() or [""]:
        logger.info(line)


def _collect_diagnostic_failures(diagnostic_lines: list[str]) -> Counter[str]:
    failures: Counter[str] = Counter()
    for line in diagnostic_lines:
        parts = [part.strip() for part in line.split(";")]
        label = parts[0]
        if " near-miss" in label:
            label = label.split(" near-miss", maxsplit=1)[0]
        elif " ready" in label:
            label = label.split(" ready", maxsplit=1)[0]
        else:
            failures.update([label])
            continue

        missing_part = next((part for part in parts[1:] if part.startswith("missing=")), None)
        if not missing_part:
            continue
        missing_checks = [item.strip() for item in missing_part.removeprefix("missing=").split(",") if item.strip()]
        failures.update(f"{label}/{check}" for check in missing_checks)
    return failures


def _enabled_timeframes(config: BotConfig) -> list[str]:
    timeframes: list[str] = []
    for pattern in config.patterns.values():
        if not pattern.enabled:
            continue
        for timeframe in pattern.settings.get("timeframes", []):
            if timeframe not in timeframes:
                timeframes.append(timeframe)
    return timeframes


def _fetch_key(symbol: str, timeframe: str) -> str:
    return f"{symbol}|{timeframe}"


def _hot_trend_score(frame: pd.DataFrame) -> float:
    if len(frame) < 50:
        return 0.0
    closes = frame["close"].astype(float)
    latest_close = float(closes.iloc[-1])
    reference_close = float(closes.iloc[-21])
    if latest_close <= 0 or reference_close <= 0:
        return 0.0
    fast_ema = float(closes.ewm(span=20, adjust=False).mean().iloc[-1])
    slow_ema = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
    directional_alignment = (latest_close >= fast_ema >= slow_ema) or (latest_close <= fast_ema <= slow_ema)
    if not directional_alignment:
        return 0.0
    return_component = min(35.0, abs(latest_close / reference_close - 1.0) * 700.0)
    separation_component = min(25.0, abs(fast_ema - slow_ema) / latest_close * 10_000.0)
    recent_volume = float(frame["volume"].iloc[-1])
    average_volume = float(frame["volume"].iloc[-21:-1].mean())
    relative_volume = recent_volume / average_volume if average_volume > 0 else 1.0
    volume_component = min(15.0, max(0.0, relative_volume - 1.0) * 15.0)
    return 25.0 + return_component + separation_component + volume_component


def _near_setup_score(config: BotConfig, frame: pd.DataFrame, timeframe: str) -> float:
    highest_score = 0.0
    for pattern in config.patterns.values():
        if not pattern.enabled or timeframe not in pattern.settings.get("timeframes", []):
            continue
        try:
            diagnostics = diagnose_pattern(pattern.name, frame, pattern.settings, timeframe)
        except (KeyError, TypeError, ValueError):
            continue
        for diagnostic in diagnostics:
            parts = {key: value for key, _, value in (part.strip().partition("=") for part in diagnostic.split(";"))}
            passed = [item.strip() for item in parts.get("passed", "").split(",") if item.strip()]
            missing = [item.strip() for item in parts.get("missing", "").split(",") if item.strip()]
            total_checks = len(passed) + len(missing)
            if total_checks:
                highest_score = max(highest_score, 100.0 * len(passed) / total_checks)
    return highest_score


def _fetch_plan(
    config: BotConfig,
    state: AlertState,
    instruments: list,
    timeframes: list[str],
    now: datetime,
) -> list[tuple[object, str]]:
    app = config.app
    budget = max(1, int(getattr(app, "historical_request_budget", 20)))
    failure_retry_seconds = max(1, int(getattr(app, "historical_failure_retry_seconds", 3_600)))
    configured_cadences = getattr(app, "timeframe_refresh_seconds", None) or {}
    core_symbols = set(getattr(config.universe, "core_symbols", [])) if config.universe.enabled else set()
    due_pairs: list[tuple[bool, int, int, float, float, datetime, int, int, int, int, object, str]] = []

    for instrument_index, instrument in enumerate(instruments):
        for timeframe_index, timeframe in enumerate(timeframes):
            key = _fetch_key(instrument.symbol, timeframe)
            last_fetched_at = state.last_fetched_at(key)
            last_failed_at = state.last_failed_at(key)
            cadence = max(1, int(configured_cadences.get(timeframe, DEFAULT_TIMEFRAME_REFRESH_SECONDS.get(timeframe, 3_600))))
            if last_fetched_at is not None and now - last_fetched_at < timedelta(seconds=cadence):
                continue
            if last_failed_at is not None and now - last_failed_at < timedelta(seconds=failure_retry_seconds):
                continue
            fetch_state_rank = 0
            reference_time = datetime.min.replace(tzinfo=now.tzinfo)
            if last_failed_at is not None and (last_fetched_at is None or last_failed_at >= last_fetched_at):
                fetch_state_rank = 2
                reference_time = last_failed_at
            elif last_fetched_at is not None:
                fetch_state_rank = 1
                reference_time = last_fetched_at
            near_setup_score = float(getattr(state, "near_setup_score", lambda _: 0.0)(key))
            hot_trend_score = float(getattr(state, "hot_trend_score", lambda _: 0.0)(instrument.symbol))
            due_pairs.append(
                (
                    instrument.symbol not in core_symbols,
                    fetch_state_rank,
                    cadence,
                    -near_setup_score,
                    -hot_trend_score,
                    reference_time,
                    int(getattr(instrument, "discovery_priority", 100)),
                    int(getattr(instrument, "discovery_rank", 999)),
                    instrument_index,
                    timeframe_index,
                    instrument,
                    timeframe,
                )
            )

    due_pairs.sort(key=lambda pair: pair[:7])
    return [(instrument, timeframe) for *_, instrument, timeframe in due_pairs[:budget]]


def _next_scan_delay_seconds(
    config: BotConfig,
    state: AlertState,
    instruments: list,
    timeframes: list[str],
    now: datetime,
) -> float:
    maximum_delay = max(1, int(config.app.scan_interval_seconds))
    failure_retry_seconds = max(1, int(getattr(config.app, "historical_failure_retry_seconds", 3_600)))
    configured_cadences = getattr(config.app, "timeframe_refresh_seconds", None) or {}
    next_due_at: datetime | None = None

    for instrument in instruments:
        for timeframe in timeframes:
            key = _fetch_key(instrument.symbol, timeframe)
            last_fetched_at = state.last_fetched_at(key)
            last_failed_at = state.last_failed_at(key)
            cadence = max(1, int(configured_cadences.get(timeframe, DEFAULT_TIMEFRAME_REFRESH_SECONDS.get(timeframe, 3_600))))
            due_at = now if last_fetched_at is None else last_fetched_at + timedelta(seconds=cadence)
            if last_failed_at is not None and (last_fetched_at is None or last_failed_at >= last_fetched_at):
                due_at = max(due_at, last_failed_at + timedelta(seconds=failure_retry_seconds))
            if next_due_at is None or due_at < next_due_at:
                next_due_at = due_at

    if next_due_at is None:
        return float(maximum_delay)
    return max(0.0, min(float(maximum_delay), (next_due_at - now).total_seconds()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-timeframe trading alert bot")
    parser.add_argument("--config", default="config/live-data.yaml", help="Path to YAML config")
    parser.add_argument("--state-file", default="data/state.json", help="Path to dedupe state file")
    parser.add_argument(
        "--account-mode",
        choices=["live", "paper", "live_gateway", "paper_gateway", "live_tws", "paper_tws"],
        help="Select the IBKR connection profile to use",
    )
    parser.add_argument("--once", action="store_true", help="Run one scan and exit")
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Log near-miss diagnostics for the latest bar when no signal is produced",
    )
    return parser.parse_args()


def run_scan(
    config: BotConfig,
    state: AlertState,
    notifier: TelegramNotifier,
    provider: MarketDataRouter,
    diagnostics: bool = False,
    review_store: AlertReviewStore | None = None,
) -> list[TradeSignal]:
    signals: list[TradeSignal] = []
    frame_cache: dict[tuple[str, str], pd.DataFrame] = {}
    failed_fetches: set[tuple[str, str]] = set()
    diagnostic_failures: Counter[str] = Counter()
    empty_frames = 0
    buying_power = provider.fetch_buying_power()
    log(f"Using IBKR BuyingPower for sizing: {buying_power:.2f}")

    instruments = active_universe(config)
    if config.universe.enabled:
        log(f"Using active universe: {len(instruments)} instruments from {config.universe.database_path} or core fallback.")
    fetch_timeframes = _enabled_timeframes(config)
    fetch_plan = _fetch_plan(config, state, instruments, fetch_timeframes, datetime.now().astimezone())
    total_fetches = len(instruments) * len(fetch_timeframes)
    log(f"Historical fetch plan: {len(fetch_plan)}/{total_fetches} due pairs within this cycle's request budget.")
    for instrument, timeframe in fetch_plan:
            cache_key = (instrument.symbol, timeframe)
            try:
                frame_cache[cache_key] = provider.fetch_ohlcv(instrument, timeframe)
                state.mark_fetched(_fetch_key(instrument.symbol, timeframe), datetime.now().astimezone())
                state.mark_near_setup_score(
                    _fetch_key(instrument.symbol, timeframe),
                    _near_setup_score(config, frame_cache[cache_key], timeframe),
                )
                if timeframe == "1h" and not frame_cache[cache_key].empty:
                    state.mark_hot_trend_score(instrument.symbol, _hot_trend_score(frame_cache[cache_key]))
            except Exception as exc:
                failed_fetches.add(cache_key)
                state.mark_fetch_failed(_fetch_key(instrument.symbol, timeframe), datetime.now().astimezone())
                log(f"Skipping {instrument.symbol} {timeframe}: {exc}")

    empty_fetches = sum(frame.empty for frame in frame_cache.values())
    log(
        "Watchlist fetch coverage: "
        f"{len(frame_cache)}/{len(fetch_plan)} planned frames returned, "
        f"{empty_fetches} empty, {len(failed_fetches)} failed."
    )

    for pattern_name, pattern in config.patterns.items():
        if not pattern.enabled:
            continue
        timeframes = pattern.settings.get("timeframes", [])
        symbol_list_name = pattern.settings.get("symbol_list")
        only_symbols = set(pattern.settings.get("only_symbols", []))
        allowed_symbols = set(config.symbol_lists.get(symbol_list_name, [])) if symbol_list_name else set()
        if only_symbols:
            allowed_symbols = only_symbols
        for instrument in instruments:
            if allowed_symbols and instrument.symbol not in allowed_symbols:
                continue
            for timeframe in timeframes:
                cache_key = (instrument.symbol, timeframe)
                if cache_key in failed_fetches or cache_key not in frame_cache:
                    continue
                frame = frame_cache[cache_key]
                if frame.empty:
                    empty_frames += 1
                    continue
                pattern_signals = list(
                    run_pattern(
                        pattern_name,
                        frame,
                        pattern.settings,
                        instrument,
                        timeframe,
                        buying_power,
                        config.app.risk_per_trade_pct,
                        config.app.position_value_pct,
                    )
                )
                if diagnostics and not pattern_signals:
                    diagnostic_lines = diagnose_pattern(
                        pattern_name,
                        frame,
                        pattern.settings,
                        timeframe,
                    )
                    diagnostic_failures.update(_collect_diagnostic_failures(diagnostic_lines))
                    log(
                        f"Diagnostic {pattern_name} {instrument.symbol} {timeframe}: "
                        + " | ".join(diagnostic_lines)
                    )
                for signal in pattern_signals:
                    if state.has_seen(signal.dedupe_key):
                        continue
                    chart_path: Path | None = None
                    try:
                        chart_path = render_signal_chart(frame, signal)
                    except Exception as exc:
                        log(f"Could not render chart for {signal.dedupe_key}: {exc}")
                    if review_store is not None:
                        review_store.record(signal, frame)
                    notifier.send(signal.to_telegram_message())
                    if chart_path is not None:
                        try:
                            notifier.send_photo(
                                chart_path,
                                f"{signal.instrument.label} {signal.timeframe} {signal.pattern} {signal.direction.upper()}",
                            )
                        except Exception as exc:
                            log(f"Could not send chart for {signal.dedupe_key}: {exc}")
                    state.remember(signal.dedupe_key)
                    signals.append(signal)

    log(
        "Scan summary: "
        f"{len(frame_cache)} successful fetches, "
        f"{len(failed_fetches)} failed fetches, "
        f"{empty_frames} empty evaluations, "
        f"{len(signals)} new signals."
    )
    if diagnostics and diagnostic_failures:
        top_blockers = ", ".join(
            f"{blocker} x{count}" for blocker, count in diagnostic_failures.most_common(5)
        )
        log(f"Diagnostic summary: top blockers this scan -> {top_blockers}")
    return signals


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(Path(args.config))
    state = AlertState(Path(args.state_file))
    notifier = TelegramNotifier(config.telegram)
    provider = MarketDataRouter(config.data_source, account_mode=args.account_mode)
    review_store = AlertReviewStore(Path("data/alert-reviews.db"))

    try:
        while True:
            signals = run_scan(config, state, notifier, provider, diagnostics=args.diagnostics, review_store=review_store)
            if signals:
                for signal in signals:
                    log(signal.to_telegram_message())
                    log("-" * 40)
            else:
                log("No new signals found.")
            if args.once:
                return
            now = datetime.now().astimezone()
            delay_seconds = _next_scan_delay_seconds(
                config,
                state,
                active_universe(config),
                _enabled_timeframes(config),
                now,
            )
            next_scan_at = now + timedelta(seconds=delay_seconds)
            log(
                f"Sleeping for {delay_seconds:.0f} seconds. "
                f"Next scan at {next_scan_at.strftime('%Y-%m-%d %H:%M:%S')}."
            )
            time.sleep(delay_seconds)
    finally:
        provider.close()


if __name__ == "__main__":
    main()