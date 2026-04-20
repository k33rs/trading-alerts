from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd

from trading_bot.config import BotConfig, load_config
from trading_bot.models import Instrument, TradeSignal
from trading_bot.patterns import run_pattern


@dataclass(slots=True)
class BacktestTrade:
    signal_time: str
    pattern: str
    direction: str
    entry: float
    stop: float
    target: float
    outcome: str
    exit_time: str | None
    exit_price: float | None
    realized_r: float | None


@dataclass(slots=True)
class BacktestSummary:
    symbol: str
    timeframe: str
    trades: list[BacktestTrade]
    slippage_bps: float = 0.0

    def to_dict(self) -> dict[str, object]:
        completed = [trade for trade in self.trades if trade.realized_r is not None]
        wins = sum(trade.outcome == "target" for trade in completed)
        losses = sum(trade.outcome == "stop" for trade in completed)
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "slippage_bps": self.slippage_bps,
            "trades": [asdict(trade) for trade in self.trades],
            "trade_count": len(self.trades),
            "completed_trade_count": len(completed),
            "wins": wins,
            "losses": losses,
            "unresolved": len(self.trades) - len(completed),
            "win_rate": wins / len(completed) if completed else 0.0,
            "total_r": sum(trade.realized_r or 0.0 for trade in completed),
        }


def load_ohlcv_csv(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    normalized = {column.lower(): column for column in frame.columns}
    time_column = next((normalized[name] for name in ("date", "datetime", "timestamp", "time") if name in normalized), None)
    required_columns = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required_columns if column not in normalized]
    if time_column is None or missing:
        raise ValueError("CSV must include a date/datetime/timestamp column and open, high, low, close, volume columns.")
    frame = frame.rename(columns={normalized[column]: column for column in required_columns})
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True)
    for column in required_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.set_index(time_column)[required_columns].dropna().sort_index()


def run_backtest(
    frame: pd.DataFrame,
    config: BotConfig,
    instrument: Instrument,
    timeframe: str,
    account_value: float,
    warmup_bars: int = 220,
    slippage_bps: float = 0.0,
) -> BacktestSummary:
    if len(frame) <= warmup_bars:
        raise ValueError(f"Need more than {warmup_bars} rows to backtest; received {len(frame)}.")
    if slippage_bps < 0:
        raise ValueError("slippage_bps cannot be negative.")
    trades: list[BacktestTrade] = []
    seen_signals: set[str] = set()
    for signal_position in range(warmup_bars - 1, len(frame) - 1):
        available = frame.iloc[: signal_position + 1]
        for pattern in config.patterns.values():
            if not pattern.enabled or timeframe not in pattern.settings.get("timeframes", []):
                continue
            for signal in run_pattern(
                pattern.name,
                available,
                pattern.settings,
                instrument,
                timeframe,
                account_value,
                config.app.risk_per_trade_pct,
                config.app.position_value_pct,
            ):
                if signal.dedupe_key in seen_signals:
                    continue
                seen_signals.add(signal.dedupe_key)
                entry_position = _signal_position(frame.index, signal.candle_time)
                trades.append(_settle_trade(signal, frame.iloc[entry_position + 1 :], slippage_bps))
    return BacktestSummary(symbol=instrument.symbol, timeframe=timeframe, trades=trades, slippage_bps=slippage_bps)


def _signal_position(index: pd.Index, candle_time: str) -> int:
    timestamp = pd.Timestamp(candle_time)
    if timestamp.tzinfo is None and getattr(index, "tz", None) is not None:
        timestamp = timestamp.tz_localize(index.tz)
    if timestamp.tzinfo is not None and getattr(index, "tz", None) is None:
        timestamp = timestamp.tz_convert(None)
    positions = index.get_indexer([timestamp])
    if positions[0] < 0:
        raise ValueError(f"Signal candle '{candle_time}' is not present in the backtest frame.")
    return int(positions[0])


def _settle_trade(signal: TradeSignal, future_bars: pd.DataFrame, slippage_bps: float = 0.0) -> BacktestTrade:
    entry = _fill_price(signal.entry, signal.direction, "entry", slippage_bps)
    stop_fill = _fill_price(signal.stop_loss, signal.direction, "exit", slippage_bps)
    target_fill = _fill_price(signal.target, signal.direction, "exit", slippage_bps)
    risk = abs(entry - stop_fill)
    for timestamp, bar in future_bars.iterrows():
        low = float(bar["low"])
        high = float(bar["high"])
        if signal.direction == "long":
            hit_stop = low <= signal.stop_loss
            hit_target = high >= signal.target
        else:
            hit_stop = high >= signal.stop_loss
            hit_target = low <= signal.target
        if hit_stop:
            return _closed_trade(signal, entry, "stop", timestamp, stop_fill, -1.0)
        if hit_target:
            realized_r = abs(target_fill - entry) / risk if risk else 0.0
            return _closed_trade(signal, entry, "target", timestamp, target_fill, realized_r)
    return BacktestTrade(
        signal_time=signal.candle_time,
        pattern=signal.pattern,
        direction=signal.direction,
        entry=entry,
        stop=signal.stop_loss,
        target=signal.target,
        outcome="unresolved",
        exit_time=None,
        exit_price=None,
        realized_r=None,
    )


def _fill_price(price: float, direction: str, phase: str, slippage_bps: float) -> float:
    multiplier = 1 + slippage_bps / 10_000
    if (direction == "long" and phase == "entry") or (direction == "short" and phase == "exit"):
        return price * multiplier
    return price / multiplier


def _closed_trade(signal: TradeSignal, entry: float, outcome: str, timestamp: pd.Timestamp, exit_price: float, realized_r: float) -> BacktestTrade:
    return BacktestTrade(
        signal_time=signal.candle_time,
        pattern=signal.pattern,
        direction=signal.direction,
        entry=entry,
        stop=signal.stop_loss,
        target=signal.target,
        outcome=outcome,
        exit_time=timestamp.isoformat(),
        exit_price=exit_price,
        realized_r=realized_r,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay trading-bot strategies against a local OHLCV CSV file")
    parser.add_argument("--csv", required=True, help="OHLCV CSV with date/datetime/timestamp, open, high, low, close, volume columns")
    parser.add_argument("--symbol", required=True, help="Configured watchlist symbol to backtest")
    parser.add_argument("--timeframe", required=True, choices=["1h", "2h", "4h", "1d", "1wk", "1mo"])
    parser.add_argument("--config", default="config/live-data.yaml", help="Path to YAML config")
    parser.add_argument("--account-value", type=float, default=100_000.0, help="Reference account value used for strategy sizing")
    parser.add_argument("--warmup-bars", type=int, default=220, help="Closed bars available before the first replayed signal")
    parser.add_argument("--slippage-bps", type=float, default=0.0, help="Adverse fill slippage in basis points per entry or exit")
    parser.add_argument("--output", help="Optional path for the JSON report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    instrument = next((item for item in config.watchlist if item.symbol == args.symbol), None)
    if instrument is None:
        raise ValueError(f"Unknown watchlist symbol '{args.symbol}'.")
    summary = run_backtest(
        load_ohlcv_csv(Path(args.csv)),
        config,
        instrument,
        args.timeframe,
        args.account_value,
        args.warmup_bars,
        args.slippage_bps,
    )
    report = summary.to_dict()
    rendered = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()