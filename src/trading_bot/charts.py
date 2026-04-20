from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

from trading_bot.models import TradeSignal


def render_signal_chart(
    frame: pd.DataFrame,
    signal: TradeSignal,
    output_dir: Path = Path("data/charts"),
    default_bars: int = 90,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_frame = _chart_frame(frame, signal, default_bars)
    if plot_frame.empty:
        raise ValueError("Cannot render chart for an empty frame")

    fig, axis = plt.subplots(figsize=(13, 7), dpi=140)
    _draw_candles(axis, plot_frame)
    _draw_signal_levels(axis, plot_frame, signal)
    _draw_structural_levels(axis, signal)
    _draw_trendlines(axis, frame, plot_frame, signal)
    _set_price_axis(axis, plot_frame, signal)

    axis.set_title(f"{signal.pattern} {signal.direction.upper()} {signal.instrument.label} {signal.timeframe}")
    axis.set_ylabel("Price")
    axis.grid(True, color="#d9d9d9", linestyle="-", linewidth=0.6, alpha=0.5)
    axis.set_axisbelow(True)
    _format_x_axis(axis, plot_frame)
    axis.legend(loc="upper left", frameon=True, fontsize=8)
    fig.tight_layout()

    file_path = output_dir / f"{_safe_filename(signal.dedupe_key)}.png"
    fig.savefig(file_path)
    plt.close(fig)
    return file_path


def _chart_frame(frame: pd.DataFrame, signal: TradeSignal, default_bars: int) -> pd.DataFrame:
    cleaned = frame.copy().dropna(subset=["open", "high", "low", "close"])
    if cleaned.empty:
        return cleaned
    end_position = _nearest_position(cleaned.index, pd.Timestamp(signal.candle_time))
    start_position = max(0, end_position - default_bars + 1)
    return cleaned.iloc[start_position : end_position + 1]


def _set_price_axis(axis, plot_frame: pd.DataFrame, signal: TradeSignal) -> None:
    values = [float(plot_frame["low"].min()), float(plot_frame["high"].max()), float(signal.entry), float(signal.stop_loss), float(signal.target)]
    management = signal.metadata.get("management")
    if isinstance(management, dict) and isinstance(management.get("checkpoint_target"), (int, float)):
        values.append(float(management["checkpoint_target"]))
    for level in _signal_chart_levels(signal):
        value = level.get("value")
        if isinstance(value, (int, float)):
            values.append(float(value))
    low = min(values)
    high = max(values)
    if high <= low:
        padding = max(abs(high) * 0.01, 1e-6)
    else:
        padding = (high - low) * 0.08
    axis.set_ylim(low - padding, high + padding)


def _draw_candles(axis, plot_frame: pd.DataFrame) -> None:
    width = 0.62
    for position, (_, candle) in enumerate(plot_frame.iterrows()):
        open_price = float(candle["open"])
        high_price = float(candle["high"])
        low_price = float(candle["low"])
        close_price = float(candle["close"])
        color = "#059669" if close_price >= open_price else "#dc2626"
        axis.vlines(position, low_price, high_price, color=color, linewidth=1.0, alpha=0.9)
        body_low = min(open_price, close_price)
        body_height = max(abs(close_price - open_price), max((high_price - low_price) * 0.02, 1e-8))
        axis.add_patch(
            Rectangle(
                (position - width / 2, body_low),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.9,
            )
        )


def _draw_signal_levels(axis, plot_frame: pd.DataFrame, signal: TradeSignal) -> None:
    levels = [
        ("Entry", signal.entry, "#2563eb", "-"),
        ("Stop", signal.stop_loss, "#dc2626", "--"),
        ("Target", signal.target, "#16a34a", "--"),
    ]
    management = signal.metadata.get("management")
    if isinstance(management, dict) and isinstance(management.get("checkpoint_target"), (int, float)):
        levels.append(("Checkpoint", float(management["checkpoint_target"]), "#9333ea", ":"))

    for label, value, color, linestyle in levels:
        axis.axhline(float(value), color=color, linestyle=linestyle, linewidth=1.2, label=f"{label} {float(value):.4f}")

    signal_position = _nearest_position(plot_frame.index, pd.Timestamp(signal.candle_time))
    master_position = _nearest_position(plot_frame.index, _master_candle_timestamp(signal))
    axis.axvspan(master_position - 0.55, master_position + 0.55, color="#f59e0b", alpha=0.14, label="Confirmed master candle")
    axis.axvline(signal_position, color="#111827", linestyle=":", linewidth=1.0, alpha=0.7, label="Signal candle")


def _draw_structural_levels(axis, signal: TradeSignal) -> None:
    for level in _signal_chart_levels(signal):
        value = level.get("value")
        role = level.get("role")
        source = level.get("source")
        if not isinstance(value, (int, float)) or role not in {"support", "resistance"}:
            continue
        color = "#0f766e" if role == "support" else "#b45309"
        source_label = str(source) if isinstance(source, str) and source else role
        axis.axhline(
            float(value),
            color=color,
            linestyle=":",
            linewidth=0.9,
            alpha=0.6,
            label=f"{source_label.title()} {float(value):.4f}",
        )


def _draw_trendlines(axis, frame: pd.DataFrame, plot_frame: pd.DataFrame, signal: TradeSignal) -> None:
    trendlines = _signal_trendlines(signal)
    if not trendlines:
        return
    for index, trendline in enumerate(trendlines):
        _draw_trendline(axis, frame, plot_frame, signal, trendline, index)


def _signal_trendlines(signal: TradeSignal) -> list[dict]:
    chart_trendlines = signal.metadata.get("chart_trendlines")
    if isinstance(chart_trendlines, list) and chart_trendlines:
        return [item for item in chart_trendlines if isinstance(item, dict)]
    trendlines = signal.metadata.get("trendlines")
    if isinstance(trendlines, list) and trendlines:
        return [item for item in trendlines if isinstance(item, dict)]
    trendline = signal.metadata.get("trendline")
    if isinstance(trendline, dict):
        return [trendline]
    return []


def _signal_chart_levels(signal: TradeSignal) -> list[dict]:
    chart_levels = signal.metadata.get("chart_levels")
    if not isinstance(chart_levels, list):
        return []
    return [level for level in chart_levels if isinstance(level, dict)]


def _master_candle_timestamp(signal: TradeSignal) -> pd.Timestamp:
    master_candle = signal.metadata.get("master_candle")
    master_time = master_candle.get("time") if isinstance(master_candle, dict) else None
    if isinstance(master_time, str) and master_time:
        return pd.Timestamp(master_time)
    return pd.Timestamp(signal.candle_time)


def _draw_trendline(axis, frame: pd.DataFrame, plot_frame: pd.DataFrame, signal: TradeSignal, trendline: dict, index: int) -> None:
    first_time = trendline.get("first_time")
    second_time = trendline.get("second_time")
    first_value = trendline.get("first_value")
    second_value = trendline.get("second_value")
    if not isinstance(first_time, str) or not isinstance(second_time, str):
        return
    if not isinstance(first_value, (int, float)) or not isinstance(second_value, (int, float)):
        return

    full_frame = frame.dropna(subset=["open", "high", "low", "close"])
    first_full_x = _nearest_position(full_frame.index, pd.Timestamp(first_time))
    second_full_x = _nearest_position(full_frame.index, pd.Timestamp(second_time))
    if first_full_x == second_full_x:
        return

    start_full_x = full_frame.index.get_loc(plot_frame.index[0])
    end_full_x = full_frame.index.get_loc(plot_frame.index[-1])
    if not isinstance(start_full_x, int) or not isinstance(end_full_x, int):
        return

    slope = (float(second_value) - float(first_value)) / (second_full_x - first_full_x)
    line_start_full_x = max(start_full_x, first_full_x)
    y_start = float(first_value) + slope * (line_start_full_x - first_full_x)
    y_end = float(first_value) + slope * (end_full_x - first_full_x)
    if not _trendline_is_visible(plot_frame, y_start, y_end):
        return
    primary = bool(trendline.get("primary", index == 0))
    role = _trendline_role(trendline)
    color = "#16a34a" if role == "support" else "#dc2626"
    linewidth = 2.2 if primary else 1.1
    linestyle = "-" if primary else "--"
    alpha = 0.95 if primary else 0.45
    long_term = bool(trendline.get("long_term"))
    prefix = "Long-term " if long_term else ""
    label = f"Primary {role}" if primary else f"{prefix}{role.title()} trendline {index}"
    axis.plot([line_start_full_x - start_full_x, len(plot_frame) - 1], [y_start, y_end], color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, label=label)

    for full_x, value in ((first_full_x, first_value), (second_full_x, second_value)):
        if start_full_x <= full_x <= end_full_x:
            axis.scatter(full_x - start_full_x, float(value), color=color, s=34 if primary else 20, alpha=alpha, zorder=5)


def _trendline_is_visible(plot_frame: pd.DataFrame, y_start: float, y_end: float) -> bool:
    price_low = float(plot_frame["low"].min())
    price_high = float(plot_frame["high"].max())
    price_range = max(price_high - price_low, 1e-8)
    padding = price_range * 0.35
    visible_low = price_low - padding
    visible_high = price_high + padding
    line_low = min(y_start, y_end)
    line_high = max(y_start, y_end)
    return line_high >= visible_low and line_low <= visible_high


def _trendline_role(trendline: dict) -> str:
    role = trendline.get("role")
    if role in {"support", "resistance"}:
        return str(role)
    pattern = str(trendline.get("patterns", ""))
    if "resistance" in pattern:
        return "resistance"
    return "support"


def _format_x_axis(axis, plot_frame: pd.DataFrame) -> None:
    tick_count = min(6, len(plot_frame))
    if tick_count <= 1:
        positions = [0]
    else:
        step = max(1, (len(plot_frame) - 1) // (tick_count - 1))
        positions = list(range(0, len(plot_frame), step))
        if positions[-1] != len(plot_frame) - 1 and len(plot_frame) - 1 - positions[-1] >= max(2, step // 2):
            positions.append(len(plot_frame) - 1)
    labels = [pd.Timestamp(plot_frame.index[position]).strftime("%Y-%m-%d") for position in positions]
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_xlim(-1, len(plot_frame) + 4)


def _nearest_position(index: pd.Index, timestamp: pd.Timestamp) -> int:
    if timestamp.tzinfo is None and getattr(index, "tz", None) is not None:
        timestamp = timestamp.tz_localize(index.tz)
    if timestamp.tzinfo is not None and getattr(index, "tz", None) is None:
        timestamp = timestamp.tz_convert(None)
    deltas = [abs(pd.Timestamp(item) - timestamp) for item in index]
    return min(range(len(index)), key=lambda position: deltas[position])


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)