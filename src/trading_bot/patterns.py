from __future__ import annotations

from typing import Iterable, NotRequired, TypedDict

import pandas as pd

from trading_bot.data_provider import confirmed_ohlcv_frame
from trading_bot.indicators import atr, average_volume, bollinger_bands, donchian_channel, ema, slope
from trading_bot.models import Instrument, TradeSignal


class TradePlan(TypedDict):
    stop: float
    target: float
    target_1: float
    target_2: float
    checkpoint: float
    rr: float
    target_note: NotRequired[str]


def evaluate_break_in(
    frame: pd.DataFrame,
    settings: dict,
    instrument: Instrument,
    timeframe: str,
    account_value: float,
    risk_pct: float,
    position_value_pct: float,
) -> list[TradeSignal]:
    trend_slope_lookback = settings.get("trend_slope_lookback", 3)
    working = frame.copy()
    working["ema_fast"] = ema(working["close"], settings["ema_fast"])
    working["ema_slow"] = ema(working["close"], settings["ema_slow"])
    working["ema_fast_slope"] = slope(working["ema_fast"], trend_slope_lookback)
    working["atr"] = atr(working, 14)
    working["avg_volume"] = average_volume(working["volume"])
    working = _add_support_resistance_levels(working, settings)
    working = _add_trend_following_context(working, settings)
    working = working.dropna()
    pivot_window = settings.get("opposing_trendline_pivot_window", 5)
    pivot_span = pivot_window * 2 + 1
    working["pivot_high"] = working["high"].where(
        working["high"] == working["high"].rolling(pivot_span, center=True).max()
    )
    working["pivot_low"] = working["low"].where(
        working["low"] == working["low"].rolling(pivot_span, center=True).min()
    )
    if len(working) < 2:
        return []

    candle = working.iloc[-1]
    previous = working.iloc[-2]
    signals: list[TradeSignal] = []

    bullish_master_candle = _master_candle_status(candle, previous, "long", settings)
    bearish_master_candle = _master_candle_status(candle, previous, "short", settings)
    bullish_confirmation = bool(bullish_master_candle["passed"])
    bearish_confirmation = bool(bearish_master_candle["passed"])
    volume_confirmed = candle["volume"] >= candle["avg_volume"] * settings["volume_ratio_min"]

    trend_slope_min = float(candle["atr"]) * settings.get("trend_slope_min_atr", 0.0)
    long_trend = candle["ema_fast"] > candle["ema_slow"] and candle["ema_fast_slope"] > trend_slope_min
    short_trend = candle["ema_fast"] < candle["ema_slow"] and candle["ema_fast_slope"] < -trend_slope_min
    support_tolerance = candle["atr"] * settings["support_resistance_atr_tolerance"]
    long_context_trend = _long_term_trend_status(candle, "long", settings)
    short_context_trend = _long_term_trend_status(candle, "short", settings)
    long_level_clearance = _support_resistance_level_status(candle, "long", settings)
    short_level_clearance = _support_resistance_level_status(candle, "short", settings)
    long_opposing_trendline = _opposing_trendline_status(working, candle, "long", settings)
    short_opposing_trendline = _opposing_trendline_status(working, candle, "short", settings)
    long_break_in_trendline = _break_in_trendline_bounce_status(working, candle, previous, "long", settings)
    short_break_in_trendline = _break_in_trendline_bounce_status(working, candle, previous, "short", settings)
    long_term_support = _break_in_long_term_stop_line(working, candle, "long", settings)
    long_term_resistance = _break_in_long_term_stop_line(working, candle, "short", settings)
    chart_trendlines = _chart_trendline_context(working, candle, settings, long_break_in_trendline, short_break_in_trendline)
    _append_trendline_metadata(chart_trendlines, long_term_support.get("trendline"))
    _append_trendline_metadata(chart_trendlines, long_term_resistance.get("trendline"))
    chart_levels = _chart_level_context(working, candle, settings)
    long_corrective_reclaim = _breakout_status_for_windows(working, candle, previous, "long", settings)
    short_corrective_reclaim = _breakout_status_for_windows(working, candle, previous, "short", settings)
    long_recent_level_test = _recent_break_in_level_test_status(working, "long", settings)
    short_recent_level_test = _recent_break_in_level_test_status(working, "short", settings)

    if long_context_trend["aligned"] and long_level_clearance["clear"] and bullish_confirmation and bool(long_break_in_trendline["bounced"]):
        entry = float(candle["close"])
        stop_candidates = [float(candle["low"]), _status_float(long_break_in_trendline, "line_value")]
        if long_term_support["found"]:
            stop_candidates.append(_status_float(long_term_support, "line_value"))
        raw_stop = min(stop_candidates)
        plan = _break_in_trade_plan(working, candle, settings, "long", entry, raw_stop)
        if plan is not None and plan["rr"] >= settings["minimum_rr"]:
            stop = plan["stop"]
            target = plan["target"]
            rr = plan["rr"]
            checkpoint_target = plan["checkpoint"]
            target_note = str(plan.get("target_note", "Target uses the nearest rolling resistance."))
            notes = [
                "Long-term context is bullish via EMA or multi-year secular trend checks.",
                "Price bounced from the ascending support trendline and the master candle closed with bullish conviction.",
                target_note,
            ]
            trendline_note = _trendline_anchor_note(long_break_in_trendline)
            if trendline_note:
                notes.insert(2, trendline_note)
            trendline_context_note = _trendline_context_note(long_break_in_trendline)
            if trendline_context_note:
                notes.insert(3, trendline_context_note)
            if long_term_support["found"]:
                notes.insert(3, str(long_term_support["note"]))
            if bullish_master_candle["uncertainty_patterns"] != "none":
                notes.insert(2, f"Warning: master candle shows uncertainty ({bullish_master_candle['uncertainty_patterns']}); treat this setup as lower conviction.")
            if bullish_master_candle["patterns"] != "none":
                notes.insert(2, f"Textbook candle pattern reinforces the setup ({bullish_master_candle['patterns']}).")
            if volume_confirmed:
                notes.insert(2, "Volume is above average and reinforces the setup.")
            signals.append(
                _build_signal(
                    pattern="break_in",
                    direction="long",
                    timeframe=timeframe,
                    instrument=instrument,
                    candle=candle,
                    entry=entry,
                    stop=stop,
                    target=target,
                    rr=rr,
                    account_value=account_value,
                    risk_pct=risk_pct,
                    position_value_pct=position_value_pct,
                    notes=notes,
                    metadata={
                        "management": {
                            "checkpoint_target": checkpoint_target,
                            "final_target": target,
                            "stop_adjustment": "If price reaches the checkpoint, trail the stop to breakeven or below the most recent retest candle while preserving reward/risk.",
                        },
                        "trendline": _single_trendline_metadata(long_break_in_trendline),
                        "trendlines": list(long_break_in_trendline.get("trendlines", [])) if isinstance(long_break_in_trendline.get("trendlines"), list) else [],
                        "chart_trendlines": chart_trendlines,
                        "chart_levels": chart_levels,
                    },
                )
            )

    if settings.get("break_in_allow_corrective_reclaim", False) and long_trend and long_context_trend["aligned"] and long_level_clearance["clear"] and bullish_confirmation and bool(long_recent_level_test["tested"]) and bool(long_corrective_reclaim["corrective_line_break"]):
        entry = float(candle["close"])
        raw_target = _corrective_break_in_target(working, candle, "long", long_corrective_reclaim, settings)
        plan = _trade_plan(candle, settings, "long", entry, float(min(candle["low"], candle["close"] - candle["atr"])), raw_target)
        if plan is not None and plan["rr"] >= settings["minimum_rr"] and not any(signal.direction == "long" for signal in signals):
            stop = plan["stop"]
            target = plan["target"]
            rr = plan["rr"]
            checkpoint_target = plan["checkpoint"]
            notes = [
                f"Short-term trend is bullish via EMA alignment and meaningful {trend_slope_lookback}-bar EMA slope; long-term context is bullish via EMA or multi-year secular trend checks.",
                "Price recently tested support, then reclaimed the corrective trendline; this can coexist with a break-out if price also clears the prior peak.",
                "Target uses the prior recent peak or the structural breakout threshold.",
            ]
            if bullish_master_candle["uncertainty_patterns"] != "none":
                notes.insert(2, f"Warning: master candle shows uncertainty ({bullish_master_candle['uncertainty_patterns']}); treat this setup as lower conviction.")
            if bullish_master_candle["patterns"] != "none":
                notes.insert(2, f"Textbook candle pattern reinforces the setup ({bullish_master_candle['patterns']}).")
            if volume_confirmed:
                notes.insert(2, "Volume is above average and reinforces the setup.")
            signals.append(
                _build_signal(
                    pattern="break_in",
                    direction="long",
                    timeframe=timeframe,
                    instrument=instrument,
                    candle=candle,
                    entry=entry,
                    stop=stop,
                    target=target,
                    rr=rr,
                    account_value=account_value,
                    risk_pct=risk_pct,
                    position_value_pct=position_value_pct,
                    notes=notes,
                    metadata={
                        "management": {
                            "checkpoint_target": checkpoint_target,
                            "final_target": target,
                            "stop_adjustment": "If price reaches the checkpoint, trail the stop to breakeven or below the most recent retest candle while preserving reward/risk.",
                        }
                    },
                )
            )

    if short_context_trend["aligned"] and short_level_clearance["clear"] and bearish_confirmation and bool(short_break_in_trendline["bounced"]):
        entry = float(candle["close"])
        stop_candidates = [float(candle["high"]), _status_float(short_break_in_trendline, "line_value")]
        if long_term_resistance["found"]:
            stop_candidates.append(_status_float(long_term_resistance, "line_value"))
        raw_stop = max(stop_candidates)
        plan = _break_in_trade_plan(working, candle, settings, "short", entry, raw_stop)
        if plan is not None and plan["rr"] >= settings["minimum_rr"]:
            stop = plan["stop"]
            target = plan["target"]
            rr = plan["rr"]
            checkpoint_target = plan["checkpoint"]
            target_note = str(plan.get("target_note", "Target uses the nearest rolling support."))
            notes = [
                "Long-term context is bearish via EMA or multi-year secular trend checks.",
                "Price rejected the descending resistance trendline and the master candle closed with bearish conviction.",
                target_note,
            ]
            trendline_note = _trendline_anchor_note(short_break_in_trendline)
            if trendline_note:
                notes.insert(2, trendline_note)
            trendline_context_note = _trendline_context_note(short_break_in_trendline)
            if trendline_context_note:
                notes.insert(3, trendline_context_note)
            if long_term_resistance["found"]:
                notes.insert(3, str(long_term_resistance["note"]))
            if bearish_master_candle["uncertainty_patterns"] != "none":
                notes.insert(2, f"Warning: master candle shows uncertainty ({bearish_master_candle['uncertainty_patterns']}); treat this setup as lower conviction.")
            if bearish_master_candle["patterns"] != "none":
                notes.insert(2, f"Textbook candle pattern reinforces the setup ({bearish_master_candle['patterns']}).")
            if volume_confirmed:
                notes.insert(2, "Volume is above average and reinforces the setup.")
            signals.append(
                _build_signal(
                    pattern="break_in",
                    direction="short",
                    timeframe=timeframe,
                    instrument=instrument,
                    candle=candle,
                    entry=entry,
                    stop=stop,
                    target=target,
                    rr=rr,
                    account_value=account_value,
                    risk_pct=risk_pct,
                    position_value_pct=position_value_pct,
                    notes=notes,
                    metadata={
                        "management": {
                            "checkpoint_target": checkpoint_target,
                            "final_target": target,
                            "stop_adjustment": "If price reaches the checkpoint, trail the stop to breakeven or above the most recent retest candle while preserving reward/risk.",
                        },
                        "trendline": _single_trendline_metadata(short_break_in_trendline),
                        "trendlines": list(short_break_in_trendline.get("trendlines", [])) if isinstance(short_break_in_trendline.get("trendlines"), list) else [],
                        "chart_trendlines": chart_trendlines,
                        "chart_levels": chart_levels,
                    },
                )
            )

    if settings.get("break_in_allow_corrective_reclaim", False) and short_trend and short_context_trend["aligned"] and short_level_clearance["clear"] and bearish_confirmation and bool(short_recent_level_test["tested"]) and bool(short_corrective_reclaim["corrective_line_break"]):
        entry = float(candle["close"])
        raw_target = _corrective_break_in_target(working, candle, "short", short_corrective_reclaim, settings)
        plan = _trade_plan(candle, settings, "short", entry, float(max(candle["high"], candle["close"] + candle["atr"])), raw_target)
        if plan is not None and plan["rr"] >= settings["minimum_rr"] and not any(signal.direction == "short" for signal in signals):
            stop = plan["stop"]
            target = plan["target"]
            rr = plan["rr"]
            checkpoint_target = plan["checkpoint"]
            notes = [
                f"Short-term trend is bearish via EMA alignment and meaningful {trend_slope_lookback}-bar EMA slope; long-term context is bearish via EMA or multi-year secular trend checks.",
                "Price recently tested resistance, then lost the corrective trendline; this can coexist with a break-out if price also clears the prior trough.",
                "Target uses the prior recent trough or the structural breakout threshold.",
            ]
            if bearish_master_candle["uncertainty_patterns"] != "none":
                notes.insert(2, f"Warning: master candle shows uncertainty ({bearish_master_candle['uncertainty_patterns']}); treat this setup as lower conviction.")
            if bearish_master_candle["patterns"] != "none":
                notes.insert(2, f"Textbook candle pattern reinforces the setup ({bearish_master_candle['patterns']}).")
            if volume_confirmed:
                notes.insert(2, "Volume is above average and reinforces the setup.")
            signals.append(
                _build_signal(
                    pattern="break_in",
                    direction="short",
                    timeframe=timeframe,
                    instrument=instrument,
                    candle=candle,
                    entry=entry,
                    stop=stop,
                    target=target,
                    rr=rr,
                    account_value=account_value,
                    risk_pct=risk_pct,
                    position_value_pct=position_value_pct,
                    notes=notes,
                    metadata={
                        "management": {
                            "checkpoint_target": checkpoint_target,
                            "final_target": target,
                            "stop_adjustment": "If price reaches the checkpoint, trail the stop to breakeven or above the most recent retest candle while preserving reward/risk.",
                        }
                    },
                )
            )

    return signals


def evaluate_break_out(
    frame: pd.DataFrame,
    settings: dict,
    instrument: Instrument,
    timeframe: str,
    account_value: float,
    risk_pct: float,
    position_value_pct: float,
) -> list[TradeSignal]:
    pivot_window = settings["pivot_window"]
    working = frame.copy()
    working["ema_fast"] = ema(working["close"], settings["ema_fast"])
    working["ema_slow"] = ema(working["close"], settings["ema_slow"])
    working["ema_fast_slope"] = slope(working["ema_fast"], settings.get("trend_slope_lookback", 3))
    working["atr"] = atr(working, settings["atr_period"])
    working = _add_trend_following_context(working, settings)
    working = working.dropna()
    pivot_span = pivot_window * 2 + 1
    working["pivot_high"] = working["high"].where(
        working["high"] == working["high"].rolling(pivot_span, center=True).max()
    )
    working["pivot_low"] = working["low"].where(
        working["low"] == working["low"].rolling(pivot_span, center=True).min()
    )
    if len(working) < 2:
        return []

    signals: list[TradeSignal] = []
    signal_window = max(1, int(settings.get("breakout_signal_valid_window_bars", settings.get("breakout_master_candle_window_bars", 3))))
    start = max(1, len(working) - signal_window)
    for direction in ("long", "short"):
        for position in range(len(working) - 1, start - 1, -1):
            candidate = _breakout_candidate_signal(
                working.iloc[: position + 1],
                settings,
                instrument,
                timeframe,
                account_value,
                risk_pct,
                position_value_pct,
                direction,
            )
            if candidate is not None:
                signals.append(candidate)
                break

    return signals


def _breakout_candidate_signal(
    working: pd.DataFrame,
    settings: dict,
    instrument: Instrument,
    timeframe: str,
    account_value: float,
    risk_pct: float,
    position_value_pct: float,
    direction: str,
) -> TradeSignal | None:
    if len(working) < 2:
        return None

    candle = working.iloc[-1]
    previous = working.iloc[-2]
    breakout = _breakout_status_for_windows(working, candle, previous, direction, settings)
    trend = _breakout_trend_status(candle, previous, direction, settings)
    context_trend = _long_term_trend_status(candle, direction, settings)
    level_clearance = _support_resistance_level_status(candle, direction, settings)
    if not (trend["passed"] and context_trend["aligned"] and level_clearance["clear"] and breakout["triggered"]):
        return None

    entry = float(candle["close"])
    extension_distance = _status_float(breakout, "extension_distance")
    if extension_distance <= 0:
        return None
    fib_levels = _breakout_fib_levels(breakout)
    if fib_levels is None:
        return None
    raw_stop = float(fib_levels["stop"])
    target_1 = float(fib_levels["target_1"])
    target_2 = float(fib_levels["target_2"])
    if direction == "long":
        trend_note = "Short-term trend is bullish via EMA alignment; long-term context is bullish via EMA or multi-year secular trend checks."
        if trend["mode"] == "emerging":
            trend_note = "Emerging bullish breakout: price reclaimed the fast EMA with positive fast-EMA slope while long-term trend context stayed constructive."
        structure_note = "Price recently broke above the descending corrective trendline and stayed inside the valid follow-through window."
        stop_adjustment = "If price reaches the checkpoint, trail the stop to breakeven or below the last impulsive retest candle while preserving reward/risk."
    else:
        trend_note = "Short-term trend is bearish via EMA alignment; long-term context is bearish via EMA or multi-year secular trend checks."
        if trend["mode"] == "emerging":
            trend_note = "Emerging bearish breakout: price lost the fast EMA with negative fast-EMA slope while long-term trend context stayed constructive."
        structure_note = "Price recently broke below the ascending corrective trendline and stayed inside the valid follow-through window."
        stop_adjustment = "If price reaches the checkpoint, trail the stop to breakeven or above the last impulsive retest candle while preserving reward/risk."

    plan = _breakout_trade_plan(candle, settings, direction, entry, raw_stop, target_1, target_2, breakout)
    if plan is None or plan["rr"] < settings["minimum_rr"]:
        return None
    target_path = _breakout_target_path_status(working, candle, direction, plan["target"], settings)
    if not target_path["clear"]:
        return None
    trendline_context = _long_term_trendline_context(working, candle, direction, plan["target"], settings)
    if not trendline_context["clear"]:
        return None
    stop = plan["stop"]
    target = plan["target"]
    rr = plan["rr"]
    checkpoint_target = plan["checkpoint"]

    target_note = str(plan.get("target_note", "Target uses a 100% projection of the deepest pullback into the correction."))
    notes = [trend_note, structure_note, target_note]
    fib_note = _fib_anchor_note(fib_levels)
    if fib_note:
        notes.insert(2, fib_note)
    if trendline_context["reinforced"]:
        notes.insert(2, str(trendline_context["reinforcement_note"]))
    if breakout["master_candle_patterns"] != "none":
        notes.insert(2, f"Textbook candle pattern reinforces the breakout ({breakout['master_candle_patterns']}).")
    if breakout["uncertainty_patterns"] != "none":
        notes.insert(2, f"Warning: breakout candle shows uncertainty ({breakout['uncertainty_patterns']}); treat this setup as lower conviction.")

    metadata: dict[str, object] = {
        "management": {
            "checkpoint_target": checkpoint_target,
            "final_target": target,
            "stop_adjustment": stop_adjustment,
        },
        "master_candle": {
            "time": str(breakout.get("master_candle_time", "")),
            "open": breakout.get("master_candle_open"),
            "close": breakout.get("master_candle_close"),
        },
    }
    trendline_metadata = _breakout_trendline_metadata(breakout, direction)
    if trendline_metadata:
        metadata["trendline"] = trendline_metadata
    metadata["fibonacci"] = fib_levels
    chart_trendlines = _breakout_chart_trendline_context(working, candle, settings, direction, trendline_metadata)
    for trendline in _long_term_chart_trendlines(working, candle, direction, plan["target"], settings):
        _append_trendline_metadata(chart_trendlines, trendline)
    if chart_trendlines:
        metadata["chart_trendlines"] = chart_trendlines
    metadata["chart_levels"] = _chart_level_context(working, candle, settings)

    return _build_signal(
        pattern="break_out",
        direction=direction,
        timeframe=timeframe,
        instrument=instrument,
        candle=candle,
        entry=entry,
        stop=stop,
        target=target,
        rr=rr,
        account_value=account_value,
        risk_pct=risk_pct,
        position_value_pct=position_value_pct,
        notes=notes,
        metadata=metadata,
    )


def evaluate_bollinger_reversion(
    frame: pd.DataFrame,
    settings: dict,
    instrument: Instrument,
    timeframe: str,
    account_value: float,
    risk_pct: float,
    position_value_pct: float,
) -> list[TradeSignal]:
    working = frame.copy()
    bands = bollinger_bands(working["close"], settings["bb_period"], settings["bb_stddev"])
    donchian = donchian_channel(working, settings["donchian_period"])
    working = working.join(bands)
    working = working.join(donchian)
    working["atr_filter"] = atr(working, settings["atr_filter_period"])
    working["atr_filter_reference"] = working["atr_filter"].shift(settings["atr_filter_reference_lookback"])
    working["donchian_upper_reference"] = working["donchian_upper"].shift(settings["donchian_reference_lookback"])
    working["donchian_lower_reference"] = working["donchian_lower"].shift(settings["donchian_reference_lookback"])
    working["momentum"] = working["close"].diff(settings["momentum_lookback"])
    working["stop_long"] = working["low"].rolling(settings["stop_lookback"]).min()
    working["stop_short"] = working["high"].rolling(settings["stop_lookback"]).max()
    working["super_open"] = working["open"].shift(settings["super_candle_lookback"] - 1)
    working["super_high"] = working["high"].rolling(settings["super_candle_lookback"]).max()
    working["super_low"] = working["low"].rolling(settings["super_candle_lookback"]).min()
    super_range = (working["super_high"] - working["super_low"]).replace(0, pd.NA)
    working["super_body_ratio"] = (working["super_open"] - working["close"]) / super_range
    working = working.dropna()
    if working.empty:
        return []

    candle = working.iloc[-1]
    previous = working.iloc[-2]
    signals: list[TradeSignal] = []
    long_filter_ok = _bollinger_long_filter(candle, settings)["passed"] or _bollinger_long_filter(previous, settings)["passed"]
    long_master_candle = _master_candle_status(candle, previous, "long", settings)
    long_band_pierce = _bollinger_lower_band_pierced(candle) or _bollinger_lower_band_pierced(previous)

    long_signal = (
        candle["close"] > candle["lower"]
        and bool(long_master_candle["passed"])
        and long_band_pierce
        and max(candle["momentum"], previous["momentum"]) > 0
        and long_filter_ok
    )
    if long_signal:
        entry = float(candle["close"])
        plan = _trade_plan(candle, settings, "long", entry, float(candle["stop_long"]), float(candle["basis"] - 0.3 * candle["deviation"]))
        if plan is not None and plan["rr"] >= _minimum_reward_risk(settings):
            stop = plan["stop"]
            target = plan["target"]
            rr = plan["rr"]
            checkpoint_target = plan["checkpoint"]
            notes = [
                "Master candle closed back above the lower Bollinger band.",
                "The current or previous confirmed candle pierced the lower band.",
                "Textbook filters were sufficiently constructive on the current or previous bar.",
                "Target uses the 100-period mean minus 0.3 standard deviations.",
            ]
            if long_master_candle["patterns"] != "none":
                notes.insert(1, f"Textbook candle pattern reinforces the setup ({long_master_candle['patterns']}).")
            if long_master_candle["uncertainty_patterns"] != "none":
                notes.insert(1, f"Warning: master candle shows uncertainty ({long_master_candle['uncertainty_patterns']}); treat this setup as lower conviction.")
            signals.append(
                _build_signal(
                    pattern="bollinger_reversion",
                    direction="long",
                    timeframe=timeframe,
                    instrument=instrument,
                    candle=candle,
                    entry=entry,
                    stop=stop,
                    target=target,
                    rr=rr,
                    account_value=account_value,
                    risk_pct=risk_pct,
                    position_value_pct=position_value_pct,
                    notes=notes,
                    metadata={
                        "management": {
                            "checkpoint_target": checkpoint_target,
                            "final_target": target,
                            "stop_adjustment": "If price reaches the checkpoint, trail the stop to breakeven or below the latest higher low while preserving reward/risk.",
                        }
                    },
                )
            )

    if settings.get("long_only", False):
        return signals

    short_signal = (
        candle["close"] < candle["upper"]
        and bool(_master_candle_status(candle, previous, "short", settings)["passed"])
        and (_bollinger_upper_band_pierced(candle) or _bollinger_upper_band_pierced(previous))
        and min(candle["momentum"], previous["momentum"]) < 0
    )
    if short_signal:
        entry = float(candle["close"])
        plan = _trade_plan(candle, settings, "short", entry, float(candle["stop_short"]), float(candle["basis"] + 0.3 * candle["deviation"]))
        if plan is not None and plan["rr"] >= _minimum_reward_risk(settings):
            stop = plan["stop"]
            target = plan["target"]
            rr = plan["rr"]
            checkpoint_target = plan["checkpoint"]
            signals.append(
                _build_signal(
                    pattern="bollinger_reversion",
                    direction="short",
                    timeframe=timeframe,
                    instrument=instrument,
                    candle=candle,
                    entry=entry,
                    stop=stop,
                    target=target,
                    rr=rr,
                    account_value=account_value,
                    risk_pct=risk_pct,
                    position_value_pct=position_value_pct,
                    notes=[
                        "Master candle closed back below the upper Bollinger band.",
                        "The current or previous candle pierced the upper band.",
                        "Target uses the 100-period mean plus 0.3 standard deviations.",
                    ],
                    metadata={
                        "management": {
                            "checkpoint_target": checkpoint_target,
                            "final_target": target,
                            "stop_adjustment": "If price reaches the checkpoint, trail the stop to breakeven or above the latest lower high while preserving reward/risk.",
                        }
                    },
                )
            )

    return signals


def run_pattern(
    name: str,
    frame: pd.DataFrame,
    settings: dict,
    instrument: Instrument,
    timeframe: str,
    account_value: float,
    risk_pct: float,
    position_value_pct: float,
) -> Iterable[TradeSignal]:
    frame = confirmed_ohlcv_frame(frame, timeframe)
    if frame.empty:
        return []
    evaluators = {
        "break_in": evaluate_break_in,
        "break_out": evaluate_break_out,
        "bollinger_reversion": evaluate_bollinger_reversion,
    }
    return evaluators[name](frame, settings, instrument, timeframe, account_value, risk_pct, position_value_pct)


def diagnose_pattern(
    name: str,
    frame: pd.DataFrame,
    settings: dict,
    timeframe: str = "1d",
) -> list[str]:
    if name == "break_out":
        return _diagnose_break_out(frame, settings, timeframe)
    diagnostics = {
        "break_in": _diagnose_break_in,
        "bollinger_reversion": _diagnose_bollinger_reversion,
    }
    return diagnostics[name](frame, settings)


def _master_candle_status(candle: pd.Series, previous: pd.Series, direction: str, settings: dict) -> dict[str, bool | str]:
    candle_range = float(candle["high"] - candle["low"])
    if candle_range <= 0:
        return {"passed": False, "patterns": "none", "uncertainty_patterns": "none"}

    open_price = float(candle["open"])
    close_price = float(candle["close"])
    high = float(candle["high"])
    low = float(candle["low"])
    previous_open = float(previous["open"])
    previous_close = float(previous["close"])
    previous_body_midpoint = (previous_open + previous_close) / 2
    body = abs(close_price - open_price)
    body_ratio = body / candle_range
    upper_wick = high - max(open_price, close_price)
    lower_wick = min(open_price, close_price) - low
    min_body_to_range = settings.get("master_candle_min_body_to_range", 0.15)
    strong_body_to_range = settings.get("master_candle_strong_body_to_range", 0.55)
    close_zone = settings.get("master_candle_close_zone", 0.25)
    uncertainty_body_max = settings.get("master_candle_uncertainty_body_max", 0.12)
    spinning_top_body_max = settings.get("master_candle_spinning_top_body_max", 0.25)
    uncertainty_wick_min = settings.get("master_candle_uncertainty_wick_min", 0.30)
    patterns: list[str] = []
    uncertainty_patterns: list[str] = []

    upper_wick_ratio = upper_wick / candle_range
    lower_wick_ratio = lower_wick / candle_range
    if body_ratio <= uncertainty_body_max:
        uncertainty_patterns.append("doji")
        if upper_wick_ratio >= uncertainty_wick_min and lower_wick_ratio >= uncertainty_wick_min:
            uncertainty_patterns.append("long_legged_doji")
    elif body_ratio <= spinning_top_body_max and upper_wick_ratio >= uncertainty_wick_min and lower_wick_ratio >= uncertainty_wick_min:
        uncertainty_patterns.append("spinning_top")

    if direction == "long":
        closes_bullish = close_price > open_price
        closes_in_upper_zone = close_price >= high - candle_range * close_zone
        directional_master = closes_bullish and body_ratio >= min_body_to_range and closes_in_upper_zone
        if lower_wick >= body * 1.5 and closes_in_upper_zone:
            patterns.append("hammer")
        if lower_wick >= candle_range * 0.5 and closes_in_upper_zone:
            patterns.append("bullish_pin_bar")
        if previous_close < previous_open and close_price > previous_open and open_price <= previous_close:
            patterns.append("bullish_engulfing")
        if previous_close < previous_open and open_price < previous_close and close_price > previous_body_midpoint:
            patterns.append("piercing_line")
        if closes_bullish and body_ratio >= strong_body_to_range and closes_in_upper_zone:
            patterns.append("bullish_marubozu_style")
    else:
        closes_bearish = close_price < open_price
        closes_in_lower_zone = close_price <= low + candle_range * close_zone
        directional_master = closes_bearish and body_ratio >= min_body_to_range and closes_in_lower_zone
        if upper_wick >= body * 1.5 and closes_in_lower_zone:
            patterns.append("shooting_star")
        if upper_wick >= candle_range * 0.5 and closes_in_lower_zone:
            patterns.append("bearish_pin_bar")
        if previous_close > previous_open and close_price < previous_open and open_price >= previous_close:
            patterns.append("bearish_engulfing")
        if previous_close > previous_open and open_price > previous_close and close_price < previous_body_midpoint:
            patterns.append("dark_cloud_cover")
        if closes_bearish and body_ratio >= strong_body_to_range and closes_in_lower_zone:
            patterns.append("bearish_marubozu_style")

    passed = directional_master
    return {
        "passed": passed,
        "patterns": ", ".join(patterns) if patterns else "none",
        "uncertainty_patterns": ", ".join(uncertainty_patterns) if uncertainty_patterns else "none",
    }


def _diagnose_break_in(
    frame: pd.DataFrame,
    settings: dict,
) -> list[str]:
    trend_slope_lookback = settings.get("trend_slope_lookback", 3)
    working = frame.copy()
    working["ema_fast"] = ema(working["close"], settings["ema_fast"])
    working["ema_slow"] = ema(working["close"], settings["ema_slow"])
    working["ema_fast_slope"] = slope(working["ema_fast"], trend_slope_lookback)
    working["atr"] = atr(working, 14)
    working = _add_support_resistance_levels(working, settings)
    working = _add_trend_following_context(working, settings)
    working = working.dropna()
    pivot_window = settings.get("opposing_trendline_pivot_window", 5)
    pivot_span = pivot_window * 2 + 1
    working["pivot_high"] = working["high"].where(
        working["high"] == working["high"].rolling(pivot_span, center=True).max()
    )
    working["pivot_low"] = working["low"].where(
        working["low"] == working["low"].rolling(pivot_span, center=True).min()
    )
    if len(working) < 2:
        return ["needs more historical bars after warmup"]

    candle = working.iloc[-1]
    previous = working.iloc[-2]
    support_tolerance = candle["atr"] * settings["support_resistance_atr_tolerance"]
    long_context_trend = _long_term_trend_status(candle, "long", settings)
    short_context_trend = _long_term_trend_status(candle, "short", settings)
    long_level_clearance = _support_resistance_level_status(candle, "long", settings)
    short_level_clearance = _support_resistance_level_status(candle, "short", settings)
    long_opposing_trendline = _opposing_trendline_status(working, candle, "long", settings)
    short_opposing_trendline = _opposing_trendline_status(working, candle, "short", settings)
    long_break_in_trendline = _break_in_trendline_bounce_status(working, candle, previous, "long", settings)
    short_break_in_trendline = _break_in_trendline_bounce_status(working, candle, previous, "short", settings)
    long_corrective_reclaim = _breakout_status_for_windows(working, candle, previous, "long", settings)
    short_corrective_reclaim = _breakout_status_for_windows(working, candle, previous, "short", settings)
    long_recent_level_test = _recent_break_in_level_test_status(working, "long", settings)
    short_recent_level_test = _recent_break_in_level_test_status(working, "short", settings)

    long_master_candle = _master_candle_status(candle, previous, "long", settings)
    long_checks = {
        "long_term_trend": bool(long_context_trend["aligned"]),
        "level_clearance": bool(long_level_clearance["clear"]),
        "trendline_bounce": bool(long_break_in_trendline["bounced"]),
        "not_uncertain": long_master_candle["uncertainty_patterns"] == "none",
        "master_candle": bool(long_master_candle["passed"]),
    }
    long_plan = _break_in_trade_plan(working, candle, settings, "long", float(candle["close"]), float(min(candle["low"], _status_float(long_break_in_trendline, "line_value"))))
    long_rr = long_plan["rr"] if long_plan is not None else 0.0
    long_corrective_checks = {
        "short_term_trend": candle["ema_fast"] > candle["ema_slow"] and candle["ema_fast_slope"] > float(candle["atr"]) * settings.get("trend_slope_min_atr", 0.0),
        "long_term_trend": bool(long_context_trend["aligned"]),
        "level_clearance": bool(long_level_clearance["clear"]),
        "recent_support_test": bool(long_recent_level_test["tested"]),
        "corrective_line_break": bool(long_corrective_reclaim["corrective_line_break"]),
        "not_uncertain": long_master_candle["uncertainty_patterns"] == "none",
        "master_candle": bool(long_master_candle["passed"]),
    }
    long_corrective_target = _corrective_break_in_target(working, candle, "long", long_corrective_reclaim, settings)
    long_corrective_plan = _trade_plan(candle, settings, "long", float(candle["close"]), float(min(candle["low"], candle["close"] - candle["atr"])), long_corrective_target)
    long_corrective_rr = long_corrective_plan["rr"] if long_corrective_plan is not None else 0.0

    short_master_candle = _master_candle_status(candle, previous, "short", settings)
    short_checks = {
        "long_term_trend": bool(short_context_trend["aligned"]),
        "level_clearance": bool(short_level_clearance["clear"]),
        "trendline_bounce": bool(short_break_in_trendline["bounced"]),
        "not_uncertain": short_master_candle["uncertainty_patterns"] == "none",
        "master_candle": bool(short_master_candle["passed"]),
    }
    short_plan = _break_in_trade_plan(working, candle, settings, "short", float(candle["close"]), float(max(candle["high"], _status_float(short_break_in_trendline, "line_value"))))
    short_rr = short_plan["rr"] if short_plan is not None else 0.0
    short_corrective_checks = {
        "short_term_trend": candle["ema_fast"] < candle["ema_slow"] and candle["ema_fast_slope"] < -float(candle["atr"]) * settings.get("trend_slope_min_atr", 0.0),
        "long_term_trend": bool(short_context_trend["aligned"]),
        "level_clearance": bool(short_level_clearance["clear"]),
        "recent_resistance_test": bool(short_recent_level_test["tested"]),
        "corrective_line_break": bool(short_corrective_reclaim["corrective_line_break"]),
        "not_uncertain": short_master_candle["uncertainty_patterns"] == "none",
        "master_candle": bool(short_master_candle["passed"]),
    }
    short_corrective_target = _corrective_break_in_target(working, candle, "short", short_corrective_reclaim, settings)
    short_corrective_plan = _trade_plan(candle, settings, "short", float(candle["close"]), float(max(candle["high"], candle["close"] + candle["atr"])), short_corrective_target)
    short_corrective_rr = short_corrective_plan["rr"] if short_corrective_plan is not None else 0.0

    return [
        _format_diagnostic("break_in long", long_checks, long_rr, settings["minimum_rr"]),
        _format_diagnostic("break_in short", short_checks, short_rr, settings["minimum_rr"]),
        _format_diagnostic("break_in corrective long", long_corrective_checks, long_corrective_rr, settings["minimum_rr"]),
        _format_diagnostic("break_in corrective short", short_corrective_checks, short_corrective_rr, settings["minimum_rr"]),
    ]


def _diagnose_break_out(
    frame: pd.DataFrame,
    settings: dict,
    timeframe: str,
) -> list[str]:
    pivot_window = settings["pivot_window"]
    working = frame.copy()
    working["ema_fast"] = ema(working["close"], settings["ema_fast"])
    working["ema_slow"] = ema(working["close"], settings["ema_slow"])
    working["ema_fast_slope"] = slope(working["ema_fast"], settings.get("trend_slope_lookback", 3))
    working["atr"] = atr(working, settings["atr_period"])
    working = _add_trend_following_context(working, settings)
    working = working.dropna()
    pivot_span = pivot_window * 2 + 1
    working["pivot_high"] = working["high"].where(
        working["high"] == working["high"].rolling(pivot_span, center=True).max()
    )
    working["pivot_low"] = working["low"].where(
        working["low"] == working["low"].rolling(pivot_span, center=True).min()
    )
    if len(working) < 2:
        return ["needs more historical bars after warmup"]

    candle = working.iloc[-1]
    previous = working.iloc[-2]
    long_breakout = _breakout_status_for_windows(working, candle, previous, "long", settings)
    short_breakout = _breakout_status_for_windows(working, candle, previous, "short", settings)
    long_trend = _breakout_trend_status(candle, previous, "long", settings)
    short_trend = _breakout_trend_status(candle, previous, "short", settings)
    long_context_trend = _long_term_trend_status(candle, "long", settings)
    short_context_trend = _long_term_trend_status(candle, "short", settings)
    long_level_clearance = _support_resistance_level_status(candle, "long", settings)
    short_level_clearance = _support_resistance_level_status(candle, "short", settings)
    long_rr = 0.0
    long_target_path_clear = True
    long_trendline_context: dict[str, bool | float | str] = {"clear": True, "reinforced": False, "reinforcement_note": "", "weakening_note": ""}
    if long_breakout["triggered"]:
        long_fib_levels = _breakout_fib_levels(long_breakout)
        if long_fib_levels is not None:
            long_target_path_clear = bool(_breakout_target_path_status(working, candle, "long", float(long_fib_levels["target_2"]), settings)["clear"])
            long_trendline_context = _long_term_trendline_context(working, candle, "long", float(long_fib_levels["target_2"]), settings)
            long_plan = _trade_plan(candle, settings, "long", float(candle["close"]), float(long_fib_levels["stop"]), float(long_fib_levels["target_1"]), float(long_fib_levels["target_2"]))
            long_rr = long_plan["rr"] if long_plan is not None else 0.0
    short_rr = 0.0
    short_target_path_clear = True
    short_trendline_context: dict[str, bool | float | str] = {"clear": True, "reinforced": False, "reinforcement_note": "", "weakening_note": ""}
    if short_breakout["triggered"]:
        short_fib_levels = _breakout_fib_levels(short_breakout)
        if short_fib_levels is not None:
            short_target_path_clear = bool(_breakout_target_path_status(working, candle, "short", float(short_fib_levels["target_2"]), settings)["clear"])
            short_trendline_context = _long_term_trendline_context(working, candle, "short", float(short_fib_levels["target_2"]), settings)
            short_plan = _trade_plan(candle, settings, "short", float(candle["close"]), float(short_fib_levels["stop"]), float(short_fib_levels["target_1"]), float(short_fib_levels["target_2"]))
            short_rr = short_plan["rr"] if short_plan is not None else 0.0

    return [
        _format_diagnostic(
            "break_out long",
            {
                "trend": bool(long_trend["passed"]),
                "long_term_trend": bool(long_context_trend["aligned"]),
                "level_clearance": bool(long_level_clearance["clear"]),
                "pivot_count": bool(long_breakout["pivot_count"]),
                "corrective_pivots": bool(long_breakout["corrective_pivots"]),
                "pivot_break": bool(long_breakout["pivot_break"]),
                "prior_extreme_clearance": bool(long_breakout["prior_extreme_clearance"]),
                "not_uncertain": not bool(long_breakout["uncertainty_candle"]),
                "candle_conviction": bool(long_breakout["candle_conviction"]),
                "confirmation_window": bool(long_breakout["confirmation_window"]),
                "entry_proximity": bool(long_breakout["entry_proximity"]),
                "structure_break": bool(long_breakout["trendline_break"]),
                "target_path_clear": bool(long_target_path_clear),
                "long_term_trendline_clear": bool(long_trendline_context["clear"]),
                "fib_extension": bool(long_breakout["fib_extension"]),
                "extension_distance": _status_float(long_breakout, "extension_distance") > 0,
                "extension_reasonable": bool(long_breakout["extension_reasonable"]),
            },
            long_rr,
            settings["minimum_rr"],
        ),
        _format_diagnostic(
            "break_out short",
            {
                "trend": bool(short_trend["passed"]),
                "long_term_trend": bool(short_context_trend["aligned"]),
                "level_clearance": bool(short_level_clearance["clear"]),
                "pivot_count": bool(short_breakout["pivot_count"]),
                "corrective_pivots": bool(short_breakout["corrective_pivots"]),
                "pivot_break": bool(short_breakout["pivot_break"]),
                "prior_extreme_clearance": bool(short_breakout["prior_extreme_clearance"]),
                "not_uncertain": not bool(short_breakout["uncertainty_candle"]),
                "candle_conviction": bool(short_breakout["candle_conviction"]),
                "confirmation_window": bool(short_breakout["confirmation_window"]),
                "entry_proximity": bool(short_breakout["entry_proximity"]),
                "structure_break": bool(short_breakout["trendline_break"]),
                "target_path_clear": bool(short_target_path_clear),
                "long_term_trendline_clear": bool(short_trendline_context["clear"]),
                "fib_extension": bool(short_breakout["fib_extension"]),
                "extension_distance": _status_float(short_breakout, "extension_distance") > 0,
                "extension_reasonable": bool(short_breakout["extension_reasonable"]),
            },
            short_rr,
            settings["minimum_rr"],
        ),
    ]


def _diagnose_bollinger_reversion(
    frame: pd.DataFrame,
    settings: dict,
) -> list[str]:
    working = frame.copy()
    bands = bollinger_bands(working["close"], settings["bb_period"], settings["bb_stddev"])
    donchian = donchian_channel(working, settings["donchian_period"])
    working = working.join(bands)
    working = working.join(donchian)
    working["atr_filter"] = atr(working, settings["atr_filter_period"])
    working["atr_filter_reference"] = working["atr_filter"].shift(settings["atr_filter_reference_lookback"])
    working["donchian_upper_reference"] = working["donchian_upper"].shift(settings["donchian_reference_lookback"])
    working["donchian_lower_reference"] = working["donchian_lower"].shift(settings["donchian_reference_lookback"])
    working["momentum"] = working["close"].diff(settings["momentum_lookback"])
    working["stop_long"] = working["low"].rolling(settings["stop_lookback"]).min()
    working["stop_short"] = working["high"].rolling(settings["stop_lookback"]).max()
    working["super_open"] = working["open"].shift(settings["super_candle_lookback"] - 1)
    working["super_high"] = working["high"].rolling(settings["super_candle_lookback"]).max()
    working["super_low"] = working["low"].rolling(settings["super_candle_lookback"]).min()
    super_range = (working["super_high"] - working["super_low"]).replace(0, pd.NA)
    working["super_body_ratio"] = (working["super_open"] - working["close"]) / super_range
    working = working.dropna()
    if len(working) < 2:
        return ["needs more historical bars after warmup"]

    candle = working.iloc[-1]
    previous = working.iloc[-2]
    long_master_candle = _master_candle_status(candle, previous, "long", settings)
    long_band_pierce = _bollinger_lower_band_pierced(candle) or _bollinger_lower_band_pierced(previous)
    long_checks = {
        "band_reclaim": candle["close"] > candle["lower"],
        "not_uncertain": long_master_candle["uncertainty_patterns"] == "none",
        "master_candle": bool(long_master_candle["passed"]),
        "band_pierce": long_band_pierce,
        "momentum": max(candle["momentum"], previous["momentum"]) > 0,
        "filter": _bollinger_long_filter(candle, settings)["passed"] or _bollinger_long_filter(previous, settings)["passed"],
    }
    long_plan = _trade_plan(candle, settings, "long", float(candle["close"]), float(candle["stop_long"]), float(candle["basis"] - 0.3 * candle["deviation"]))
    long_rr = long_plan["rr"] if long_plan is not None else 0.0
    diagnostics = [_format_diagnostic("bollinger_reversion long", long_checks, long_rr, 0.0)]
    if settings.get("long_only", False):
        return diagnostics

    short_checks = {
        "band_reclaim": candle["close"] < candle["upper"],
        "latest_bar_strength": candle["close"] < candle["open"],
        "band_pierce": _bollinger_upper_band_pierced(candle) or _bollinger_upper_band_pierced(previous),
        "momentum": min(candle["momentum"], previous["momentum"]) < 0,
    }
    short_plan = _trade_plan(candle, settings, "short", float(candle["close"]), float(candle["stop_short"]), float(candle["basis"] + 0.3 * candle["deviation"]))
    short_rr = short_plan["rr"] if short_plan is not None else 0.0
    diagnostics.append(_format_diagnostic("bollinger_reversion short", short_checks, short_rr, 0.0))
    return diagnostics


def _format_diagnostic(label: str, checks: dict[str, bool], rr: float, minimum_rr: float) -> str:
    missing = [name for name, passed in checks.items() if not passed]
    passed = [name for name, is_ok in checks.items() if is_ok]
    status = "ready" if not missing and rr >= minimum_rr else "near-miss"
    parts: list[str] = [f"{label} {status}"]
    if passed:
        parts.append(f"passed={', '.join(passed)}")
    if missing:
        parts.append(f"missing={', '.join(missing)}")
    if minimum_rr > 0:
        parts.append(f"rr={rr:.2f}/{minimum_rr:.2f}")
    else:
        parts.append(f"rr={rr:.2f}")
    return "; ".join(parts)


def _status_float(status: dict[str, bool | float | str], key: str) -> float:
    value = status[key]
    if isinstance(value, bool) or isinstance(value, str):
        return 0.0
    return float(value)


def _bollinger_long_filter(candle: pd.Series, settings: dict) -> dict[str, bool | int]:
    atr_ok = candle["atr_filter"] <= candle["atr_filter_reference"] * settings["atr_filter_max_multiplier"]
    donchian_uptrend = (
        candle["donchian_upper"] > candle["donchian_upper_reference"]
        and candle["donchian_lower"] > candle["donchian_lower_reference"]
    )
    super_candle_ok = candle["super_body_ratio"] >= settings["super_candle_body_to_range_min"]
    checks = [bool(atr_ok), not bool(donchian_uptrend), bool(super_candle_ok)]
    passed_checks = sum(checks)
    minimum_checks = int(settings.get("minimum_filter_checks", 2))
    return {"passed": passed_checks >= minimum_checks, "passed_checks": passed_checks, "minimum_checks": minimum_checks}


def _bollinger_lower_band_pierced(candle: pd.Series) -> bool:
    return float(candle["low"]) <= float(candle["lower"])


def _bollinger_upper_band_pierced(candle: pd.Series) -> bool:
    return float(candle["high"]) >= float(candle["upper"])


def _minimum_reward_risk(settings: dict) -> float:
    return float(settings.get("minimum_rr", 1.5))


def _breakout_trend_status(candle: pd.Series, previous: pd.Series, direction: str, settings: dict) -> dict[str, bool | str]:
    trend_mode = settings.get("trend_mode", "aligned")
    if direction == "long":
        aligned = candle["ema_fast"] > candle["ema_slow"]
        emerging = candle["close"] > candle["ema_fast"] and candle["ema_fast_slope"] > 0 and candle["close"] > previous["close"]
    else:
        aligned = candle["ema_fast"] < candle["ema_slow"]
        emerging = candle["close"] < candle["ema_fast"] and candle["ema_fast_slope"] < 0 and candle["close"] < previous["close"]

    if aligned:
        return {"passed": True, "mode": "aligned"}
    if trend_mode == "aligned_or_emerging" and emerging:
        return {"passed": True, "mode": "emerging"}
    return {"passed": False, "mode": "none"}


def _recent_master_candle_status(
    working: pd.DataFrame,
    recent: pd.DataFrame,
    direction: str,
    settings: dict,
) -> dict[str, bool | str]:
    lookback = max(1, int(settings.get("breakout_master_candle_window_bars", settings.get("breakout_window_bars", 5))))
    window = recent.tail(lookback)
    uncertainty_patterns: list[str] = []
    for index, row in reversed(list(window.iterrows())):
        position = working.index.get_loc(index)
        if not isinstance(position, int) or position <= 0:
            continue
        previous = working.iloc[position - 1]
        status = _master_candle_status(row, previous, direction, settings)
        if status["passed"]:
            status["candle_time"] = pd.Timestamp(index).isoformat()
            status["open"] = float(row["open"])
            status["close"] = float(row["close"])
            return status
        if status["uncertainty_patterns"] != "none":
            uncertainty_patterns.append(str(status["uncertainty_patterns"]))

    return {
        "passed": False,
        "patterns": "none",
        "uncertainty_patterns": ", ".join(uncertainty_patterns) if uncertainty_patterns else "none",
        "candle_time": "",
    }


def _configured_lookbacks(settings: dict, windows_key: str, fallback_key: str, default: int) -> list[int | None]:
    raw_windows = settings.get(windows_key, [settings.get(fallback_key, default)])
    if not isinstance(raw_windows, list):
        raw_windows = [raw_windows]

    lookbacks: list[int | None] = []
    for raw_window in raw_windows:
        if isinstance(raw_window, str) and raw_window.lower() == "all":
            lookback = None
        else:
            lookback = int(raw_window)
            if lookback <= 0:
                lookback = None
        if lookback not in lookbacks:
            lookbacks.append(lookback)
    return lookbacks or [default]


def _lookback_label(lookback: int | None) -> str:
    return "all" if lookback is None else str(lookback)


def _rolling_level(series: pd.Series, lookback: int | None, method: str) -> pd.Series:
    shifted = series.shift(1)
    rolling = shifted.expanding(min_periods=1) if lookback is None else shifted.rolling(lookback, min_periods=1)
    return rolling.min() if method == "min" else rolling.max()


def _add_support_resistance_levels(working: pd.DataFrame, settings: dict) -> pd.DataFrame:
    context = working.copy()
    support_columns: list[str] = []
    resistance_columns: list[str] = []
    for lookback in _configured_lookbacks(settings, "support_resistance_lookbacks", "support_resistance_lookback", 20):
        label = _lookback_label(lookback)
        support_column = f"support_{label}"
        resistance_column = f"resistance_{label}"
        context[support_column] = _rolling_level(context["low"], lookback, "min")
        context[resistance_column] = _rolling_level(context["high"], lookback, "max")
        support_columns.append(support_column)
        resistance_columns.append(resistance_column)
    context["support"] = context[support_columns].max(axis=1)
    context["resistance"] = context[resistance_columns].min(axis=1)
    return context


def _recent_break_in_level_test_status(working: pd.DataFrame, direction: str, settings: dict) -> dict[str, bool | float | str]:
    lookback = max(1, int(settings.get("break_in_level_test_window_bars", 5)))
    window = working.tail(lookback)
    if window.empty:
        return {"tested": False, "level": 0.0, "distance": 0.0, "candle_time": ""}

    tolerance_multiplier = settings["support_resistance_atr_tolerance"]
    for index, row in reversed(list(window.iterrows())):
        tolerance = float(row["atr"]) * tolerance_multiplier
        if direction == "long":
            level = float(row["support"])
            distance = float(row["close"] - level)
            tested = float(row["low"]) <= level + tolerance and float(row["close"]) >= level - tolerance
        else:
            level = float(row["resistance"])
            distance = float(level - row["close"])
            tested = float(row["high"]) >= level - tolerance and float(row["close"]) <= level + tolerance
        if tested:
            return {"tested": True, "level": level, "distance": distance, "candle_time": pd.Timestamp(index).isoformat()}
    return {"tested": False, "level": 0.0, "distance": 0.0, "candle_time": ""}


def _add_trend_following_context(working: pd.DataFrame, settings: dict) -> pd.DataFrame:
    context = working.copy()
    ema_long_period = settings.get("ema_long", settings["ema_slow"] * 2)
    context["ema_long"] = ema(context["close"], ema_long_period)
    context["ema_long_slope"] = slope(context["ema_long"], settings.get("long_term_trend_slope_lookback", 10))
    secular_scores: list[pd.Series] = []
    for lookback in _configured_lookbacks(settings, "secular_trend_lookbacks", "secular_trend_lookback", 520):
        if lookback is None:
            score = (context["close"] - float(context["close"].iloc[0])) / context["atr"]
        else:
            reference = context["close"].shift(lookback)
            score = (context["close"] - reference) / context["atr"]
        secular_scores.append(score)
    if secular_scores:
        context["secular_trend_score"] = pd.concat(secular_scores, axis=1).mean(axis=1)
    else:
        context["secular_trend_score"] = 0.0
    support_columns: list[str] = []
    resistance_columns: list[str] = []
    for lookback in _configured_lookbacks(settings, "long_term_level_lookbacks", "long_term_level_lookback", settings.get("opposing_trendline_lookback", 220)):
        label = _lookback_label(lookback)
        support_column = f"long_term_support_{label}"
        resistance_column = f"long_term_resistance_{label}"
        context[support_column] = _rolling_level(context["low"], lookback, "min")
        context[resistance_column] = _rolling_level(context["high"], lookback, "max")
        support_columns.append(support_column)
        resistance_columns.append(resistance_column)
    context["long_term_support"] = context[support_columns].max(axis=1)
    context["long_term_resistance"] = context[resistance_columns].min(axis=1)
    return context


def _long_term_trend_status(candle: pd.Series, direction: str, settings: dict) -> dict[str, bool]:
    slope_min = float(candle["atr"]) * settings.get("long_term_trend_slope_min_atr", 0.02)
    secular_min = settings.get("secular_trend_min_score", 0.0)
    secular_score = float(candle.get("secular_trend_score", 0.0))
    secular_decisive = abs(secular_score) > secular_min
    if direction == "long":
        ema_aligned = candle["close"] > candle["ema_long"] and candle["ema_slow"] > candle["ema_long"] and candle["ema_long_slope"] > slope_min
        secular_aligned = secular_score > secular_min
    else:
        ema_aligned = candle["close"] < candle["ema_long"] and candle["ema_slow"] < candle["ema_long"] and candle["ema_long_slope"] < -slope_min
        secular_aligned = secular_score < -secular_min
    aligned = secular_aligned if secular_decisive else ema_aligned
    return {"aligned": bool(aligned), "ema_aligned": bool(ema_aligned), "secular_aligned": bool(secular_aligned), "secular_score": secular_score}


def _support_resistance_level_status(candle: pd.Series, direction: str, settings: dict) -> dict[str, bool | float]:
    tolerance = float(candle["atr"]) * settings.get("long_term_level_atr_tolerance", 0.75)
    close_price = float(candle["close"])
    if direction == "long":
        levels = _finite_candle_levels(candle, "long_term_resistance") or [float(candle["long_term_resistance"])]
        blocking = [(level, level - close_price) for level in levels if 0 <= level - close_price <= tolerance]
        candidates = [(level, level - close_price) for level in levels if level >= close_price]
    else:
        levels = _finite_candle_levels(candle, "long_term_support") or [float(candle["long_term_support"])]
        blocking = [(level, close_price - level) for level in levels if 0 <= close_price - level <= tolerance]
        candidates = [(level, close_price - level) for level in levels if level <= close_price]

    if blocking:
        level, distance = min(blocking, key=lambda item: item[1])
        return {"clear": False, "level": level, "distance": distance}
    if candidates:
        level, distance = min(candidates, key=lambda item: item[1])
        return {"clear": True, "level": level, "distance": distance}
    level = min(levels, key=lambda value: abs(value - close_price))
    return {"clear": True, "level": level, "distance": abs(level - close_price)}


def _finite_candle_levels(candle: pd.Series, prefix: str) -> list[float]:
    levels: list[float] = []
    for name, value in candle.items():
        if name == prefix or str(name).startswith(f"{prefix}_"):
            if pd.notna(value):
                levels.append(float(value))
    return list(dict.fromkeys(levels))


def _opposing_trendline_status(working: pd.DataFrame, candle: pd.Series, direction: str, settings: dict) -> dict[str, bool | float | str]:
    min_span = settings.get("opposing_trendline_min_span_bars", 40)
    tolerance = float(candle["atr"]) * settings.get("opposing_trendline_atr_tolerance", 0.75)
    current_x = working.index.get_loc(candle.name)
    if not isinstance(current_x, int):
        return {"clear": True, "line_value": 0.0, "distance": 0.0, "patterns": "none"}

    pivot_column = "pivot_high" if direction == "long" else "pivot_low"
    nearest_distance: float | None = None
    nearest_line_value = 0.0
    for lookback in _configured_lookbacks(settings, "opposing_trendline_lookbacks", "opposing_trendline_lookback", 220):
        recent = working if lookback is None else working.iloc[-lookback:]
        pivots = recent[recent[pivot_column].notna()][pivot_column]
        if len(pivots) < 2:
            continue
        for first_position in range(len(pivots) - 1):
            for second_position in range(first_position + 1, len(pivots)):
                if second_position != first_position + 1:
                    continue
                first_index = pivots.index[first_position]
                second_index = pivots.index[second_position]
                first_x = working.index.get_loc(first_index)
                second_x = working.index.get_loc(second_index)
                if not isinstance(first_x, int) or not isinstance(second_x, int) or second_x - first_x < min_span:
                    continue
                first_y = float(pivots.iloc[first_position])
                second_y = float(pivots.iloc[second_position])
                if direction == "long" and second_y <= first_y:
                    continue
                if direction == "short" and second_y >= first_y:
                    continue
                if _trendline_has_middle_pierce(working, first_x, second_x, first_y, second_y, current_x, "short" if direction == "long" else "long", include_current=False):
                    continue
                if not _trendline_has_later_tap(
                    working,
                    first_x,
                    second_x,
                    first_y,
                    second_y,
                    current_x,
                    "short" if direction == "long" else "long",
                    settings.get("opposing_trendline_atr_tolerance", 0.75),
                    include_current=True,
                ):
                    continue
                line_value = _project_line_value(first_x, first_y, second_x, second_y, current_x)
                if line_value is None or line_value <= 0:
                    continue
                distance = abs(float(candle["close"]) - line_value)
                if nearest_distance is None or distance < nearest_distance:
                    nearest_distance = distance
                    nearest_line_value = line_value

    if nearest_distance is None:
        return {"clear": True, "line_value": 0.0, "distance": 0.0, "patterns": "none"}
    return {
        "clear": nearest_distance > tolerance,
        "line_value": nearest_line_value,
        "distance": nearest_distance,
        "patterns": "descending_resistance" if direction == "long" else "ascending_support",
    }


def _break_in_trendline_bounce_status(
    working: pd.DataFrame,
    candle: pd.Series,
    previous: pd.Series,
    direction: str,
    settings: dict,
) -> dict[str, bool | float | str]:
    min_span = settings.get("break_in_trendline_min_span_bars", settings.get("secondary_trendline_min_span_bars", settings.get("pivot_window", 5) * 2))
    atr_tolerance_multiplier = settings.get("break_in_trendline_atr_tolerance", settings.get("secondary_trendline_atr_tolerance", 0.75))
    tolerance = float(candle["atr"]) * atr_tolerance_multiplier
    current_x = working.index.get_loc(candle.name)
    if not isinstance(current_x, int):
        return {"bounced": False, "line_value": 0.0, "distance": 0.0, "patterns": "none"}

    pivot_column = "pivot_low" if direction == "long" else "pivot_high"
    best_status = {"bounced": False, "line_value": 0.0, "distance": 0.0, "patterns": "none", "trendlines": []}
    best_rank: tuple[int, float] | None = None
    candidate_trendlines: list[dict[str, object]] = []
    recent_anchor_lookback_raw = settings.get("break_in_recent_pivot_lookback", settings.get("secondary_recent_pivot_lookback"))
    recent_anchor_lookback = int(recent_anchor_lookback_raw) if recent_anchor_lookback_raw is not None else None
    max_span_raw = settings.get("break_in_trendline_max_span_bars", settings.get("secondary_trendline_max_span_bars"))
    max_span = int(max_span_raw) if max_span_raw is not None else None
    min_anchor_move = float(candle["atr"]) * settings.get("break_in_trendline_min_anchor_move_atr", settings.get("secondary_trendline_min_anchor_move_atr", 0.75))
    anchor_cluster_bars = int(settings.get("break_in_trendline_anchor_cluster_bars", 0))
    for lookback in _chart_trendline_lookbacks(settings):
        recent = working if lookback is None else working.iloc[-lookback:]
        pivots = recent[recent[pivot_column].notna()][pivot_column]
        if anchor_cluster_bars > 0:
            pivots = _confirmed_reaction_pivots(working, pivots, anchor_cluster_bars, "low" if direction == "long" else "high")
        if len(pivots) < 2:
            continue
        for first_position in range(len(pivots) - 1):
            for second_position in range(first_position + 1, len(pivots)):
                if second_position != first_position + 1:
                    continue
                first_index = pivots.index[first_position]
                second_index = pivots.index[second_position]
                first_x = working.index.get_loc(first_index)
                second_x = working.index.get_loc(second_index)
                if not isinstance(first_x, int) or not isinstance(second_x, int) or second_x - first_x < min_span:
                    continue
                if max_span is not None and second_x - first_x > max_span:
                    continue
                if recent_anchor_lookback is not None and current_x - second_x > recent_anchor_lookback:
                    continue
                first_y = float(pivots.iloc[first_position])
                second_y = float(pivots.iloc[second_position])
                if direction == "long" and second_y <= first_y:
                    continue
                if direction == "short" and second_y >= first_y:
                    continue
                if abs(second_y - first_y) < min_anchor_move:
                    continue
                if _trendline_has_middle_pierce(working, first_x, second_x, first_y, second_y, current_x, direction, include_current=False):
                    continue
                line_value = _project_line_value(first_x, first_y, second_x, second_y, current_x)
                if line_value is None or line_value <= 0:
                    continue
                if direction == "long":
                    distance = abs(float(candle["low"]) - line_value)
                    touched = line_value - tolerance <= float(candle["low"]) <= line_value + tolerance
                    rejected = float(candle["close"]) > line_value and (float(candle["close"]) > float(candle["open"]) or float(candle["close"]) > float(previous["close"]))
                    pattern = "ascending_support_bounce"
                else:
                    distance = abs(float(candle["high"]) - line_value)
                    touched = line_value - tolerance <= float(candle["high"]) <= line_value + tolerance
                    rejected = float(candle["close"]) < line_value and float(candle["close"]) < float(candle["open"])
                    pattern = "descending_resistance_rejection"
                if not (touched and rejected):
                    continue
                cleanliness = _trendline_cleanliness(
                    working,
                    first_x,
                    second_x,
                    first_y,
                    second_y,
                    current_x,
                    direction,
                    atr_tolerance_multiplier,
                    include_current=False,
                )
                candidate_status = {
                    "bounced": True,
                    "line_value": float(line_value),
                    "distance": float(distance),
                    "patterns": pattern,
                    "first_time": pd.Timestamp(first_index).isoformat(),
                    "second_time": pd.Timestamp(second_index).isoformat(),
                    "first_value": first_y,
                    "second_value": second_y,
                    "lookback": _lookback_label(lookback),
                    "cleanliness_score": cleanliness["cleanliness_score"],
                    "close_breaches": cleanliness["close_breaches"],
                    "wick_breaches": cleanliness["wick_breaches"],
                }
                candidate_trendlines.append(candidate_status)
                rank = (int(cleanliness["cleanliness_score"]), float(distance))
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_status = dict(candidate_status)
    if bool(best_status["bounced"]):
        best_status["trendlines"] = _ranked_trendline_candidates(candidate_trendlines, best_status, settings)
    return best_status


def _trendline_has_later_tap(
    working: pd.DataFrame,
    first_x: int,
    second_x: int,
    first_y: float,
    second_y: float,
    current_x: int,
    direction: str,
    atr_tolerance_multiplier: float,
    *,
    include_current: bool,
) -> bool:
    final_x = current_x + 1 if include_current else current_x
    for row_x in range(second_x + 1, final_x):
        line_value = _project_line_value(first_x, first_y, second_x, second_y, row_x)
        if line_value is None:
            continue
        row = working.iloc[row_x]
        row_atr = row.get("atr")
        tolerance = 0.0
        if pd.notna(row_atr):
            row_atr_value = float(row_atr)
            if row_atr_value > 0:
                tolerance = row_atr_value * atr_tolerance_multiplier
        if direction == "long" and line_value - tolerance <= float(row["low"]) <= line_value + tolerance:
            return True
        if direction == "short" and line_value - tolerance <= float(row["high"]) <= line_value + tolerance:
            return True
    return False


def _trendline_has_middle_pierce(
    working: pd.DataFrame,
    first_x: int,
    second_x: int,
    first_y: float,
    second_y: float,
    current_x: int,
    direction: str,
    *,
    include_current: bool,
) -> bool:
    final_x = current_x + 1 if include_current else current_x
    for row_x in range(first_x + 1, final_x):
        line_value = _project_line_value(first_x, first_y, second_x, second_y, row_x)
        if line_value is None:
            continue
        row = working.iloc[row_x]
        if direction == "long" and float(row["low"]) < line_value:
            return True
        if direction == "short" and float(row["high"]) > line_value:
            return True
    return False


def _trendline_cleanliness(
    working: pd.DataFrame,
    first_x: int,
    second_x: int,
    first_y: float,
    second_y: float,
    current_x: int,
    direction: str,
    atr_tolerance_multiplier: float,
    *,
    include_current: bool,
) -> dict[str, int]:
    final_x = current_x + 1 if include_current else current_x
    close_breaches = 0
    wick_breaches = 0
    for row_x in range(second_x + 1, final_x):
        line_value = _project_line_value(first_x, first_y, second_x, second_y, row_x)
        if line_value is None:
            continue
        row = working.iloc[row_x]
        row_atr = row.get("atr")
        tolerance = 0.0
        if pd.notna(row_atr):
            row_atr_value = float(row_atr)
            if row_atr_value > 0:
                tolerance = row_atr_value * atr_tolerance_multiplier
        if direction == "long":
            if float(row["close"]) < line_value - tolerance:
                close_breaches += 1
            if float(row["low"]) < line_value - tolerance:
                wick_breaches += 1
        else:
            if float(row["close"]) > line_value + tolerance:
                close_breaches += 1
            if float(row["high"]) > line_value + tolerance:
                wick_breaches += 1
    return {
        "cleanliness_score": close_breaches * 3 + wick_breaches,
        "close_breaches": close_breaches,
        "wick_breaches": wick_breaches,
    }


def _confirmed_reaction_pivots(working: pd.DataFrame, pivots: pd.Series, cluster_bars: int, pivot_kind: str = "last") -> pd.Series:
    if len(pivots) < 2:
        return pivots
    confirmed: list[tuple[pd.Timestamp, float]] = []
    cluster: list[tuple[pd.Timestamp, float]] = []
    previous_x: int | None = None
    for index, value in pivots.items():
        current_x = working.index.get_loc(index)
        if not isinstance(current_x, int):
            continue
        if previous_x is None or current_x - previous_x <= cluster_bars:
            cluster.append((pd.Timestamp(index), float(value)))
        else:
            confirmed.append(_select_reaction_pivot(cluster, pivot_kind))
            cluster = [(pd.Timestamp(index), float(value))]
        previous_x = current_x
    if cluster:
        confirmed.append(_select_reaction_pivot(cluster, pivot_kind))
    return pd.Series({index: value for index, value in confirmed})


def _select_reaction_pivot(cluster: list[tuple[pd.Timestamp, float]], pivot_kind: str) -> tuple[pd.Timestamp, float]:
    if pivot_kind == "high":
        return max(cluster, key=lambda item: (item[1], item[0]))
    if pivot_kind == "low":
        return min(cluster, key=lambda item: (item[1], -item[0].value))
    return cluster[-1]


def _ranked_trendline_candidates(candidates: list[dict[str, object]], primary: dict[str, object], settings: dict) -> list[dict[str, object]]:
    max_lines = max(1, int(settings.get("break_in_trendline_max_context_lines", 4)))
    ranked: list[dict[str, object]] = []
    seen: set[tuple[object, object, object]] = set()
    primary_key = (primary.get("first_time"), primary.get("second_time"), primary.get("patterns"))
    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            int(item.get("cleanliness_score", 0)) if isinstance(item.get("cleanliness_score"), (int, float)) else 0,
            float(item.get("distance", 0.0)) if isinstance(item.get("distance"), (int, float)) else 0.0,
        ),
    )
    for candidate in [primary, *sorted_candidates]:
        key = (candidate.get("first_time"), candidate.get("second_time"), candidate.get("patterns"))
        if key in seen:
            continue
        seen.add(key)
        trendline = dict(candidate)
        trendline["primary"] = key == primary_key
        ranked.append(trendline)
        if len(ranked) >= max_lines:
            break
    return ranked


def _chart_trendline_context(working: pd.DataFrame, candle: pd.Series, settings: dict, *primary_statuses: dict[str, object]) -> list[dict[str, object]]:
    max_per_role = max(1, int(settings.get("break_in_chart_trendline_max_lines_per_side", 3)))
    max_distance = float(candle["atr"]) * settings.get("break_in_chart_trendline_max_distance_atr", 8.0)
    chart_lines: list[dict[str, object]] = []
    seen: set[tuple[object, object, object]] = set()

    for status in primary_statuses:
        if not bool(status.get("bounced")):
            continue
        trendline = _single_trendline_metadata(status)
        trendline["primary"] = True
        trendline["role"] = _trendline_role(trendline)
        key = _trendline_key(trendline)
        if key not in seen:
            seen.add(key)
            chart_lines.append(trendline)

    for role, direction in (("support", "long"), ("resistance", "short")):
        added_for_role = sum(1 for trendline in chart_lines if trendline.get("role") == role)
        for candidate in _trendline_context_candidates(working, candle, settings, direction, max_distance):
            key = _trendline_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            chart_lines.append(candidate)
            added_for_role += 1
            if added_for_role >= max_per_role:
                break
    return chart_lines


def _breakout_chart_trendline_context(working: pd.DataFrame, candle: pd.Series, settings: dict, direction: str, primary_trendline: dict[str, object] | None) -> list[dict[str, object]]:
    max_per_role = max(1, int(settings.get("breakout_chart_trendline_max_lines_per_side", settings.get("break_in_chart_trendline_max_lines_per_side", 3))))
    max_distance = float(candle["atr"]) * settings.get("breakout_chart_trendline_max_distance_atr", settings.get("break_in_chart_trendline_max_distance_atr", 8.0))
    chart_lines: list[dict[str, object]] = []
    seen: set[tuple[object, object, object]] = set()

    if primary_trendline:
        primary = dict(primary_trendline)
        primary["primary"] = True
        primary["role"] = primary.get("role") or ("resistance" if direction == "long" else "support")
        seen.add(_trendline_key(primary))
        chart_lines.append(primary)

    for role, context_direction in (("support", "long"), ("resistance", "short")):
        added_for_role = sum(1 for trendline in chart_lines if trendline.get("role") == role)
        for candidate in _trendline_context_candidates(working, candle, settings, context_direction, max_distance):
            key = _trendline_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            chart_lines.append(candidate)
            added_for_role += 1
            if added_for_role >= max_per_role:
                break
    return chart_lines


def _append_trendline_metadata(chart_lines: list[dict[str, object]], trendline: object) -> None:
    if not isinstance(trendline, dict):
        return
    key = _trendline_key(trendline)
    if any(_trendline_key(existing) == key for existing in chart_lines):
        return
    chart_lines.append(dict(trendline))


def _long_term_chart_trendlines(
    working: pd.DataFrame,
    candle: pd.Series,
    direction: str,
    target: float,
    settings: dict,
) -> list[dict[str, object]]:
    min_span = int(settings.get("long_term_trendline_min_span_bars", 80))
    max_distance = float(candle["atr"]) * settings.get("long_term_trendline_chart_max_distance_atr", 12.0)
    entry = float(candle["close"])
    candidates: list[dict[str, object]] = []
    for context_direction in ("long", "short"):
        for candidate in _trendline_context_candidates(working, candle, settings, context_direction, max_distance):
            if int(candidate.get("span_bars", 0)) < min_span:
                continue
            line_value = candidate.get("line_value")
            if not isinstance(line_value, (int, float)):
                continue
            relevant = (
                min(entry, target) <= float(line_value) <= max(entry, target)
                or abs(entry - float(line_value)) <= max_distance
            )
            if relevant:
                candidate["long_term"] = True
                candidates.append(candidate)
    return sorted(candidates, key=lambda item: float(item.get("distance", 0.0)))


def _chart_level_context(working: pd.DataFrame, candle: pd.Series, settings: dict) -> list[dict[str, object]]:
    entry = float(candle["close"])
    max_distance = float(candle["atr"]) * settings.get("chart_horizontal_level_max_distance_atr", 12.0)
    max_per_role = max(1, int(settings.get("chart_horizontal_level_max_per_side", 4)))
    candidates: list[tuple[float, str, str]] = []
    for prefix, role, source in (
        ("support", "support", "rolling support"),
        ("long_term_support", "support", "long-term support"),
        ("resistance", "resistance", "rolling resistance"),
        ("long_term_resistance", "resistance", "long-term resistance"),
    ):
        for value in _finite_candle_levels(candle, prefix):
            if abs(entry - value) <= max_distance:
                candidates.append((value, role, source))

    levels: list[dict[str, object]] = []
    for role in ("support", "resistance"):
        role_candidates = sorted(
            (candidate for candidate in candidates if candidate[1] == role),
            key=lambda candidate: abs(entry - candidate[0]),
        )
        seen: set[float] = set()
        for value, _, source in role_candidates:
            rounded = round(value, 8)
            if rounded in seen:
                continue
            seen.add(rounded)
            levels.append({"value": value, "role": role, "source": source})
            if sum(level["role"] == role for level in levels) >= max_per_role:
                break
    return levels


def _trendline_context_candidates(working: pd.DataFrame, candle: pd.Series, settings: dict, direction: str, max_distance: float) -> list[dict[str, object]]:
    min_span = settings.get("break_in_trendline_min_span_bars", settings.get("secondary_trendline_min_span_bars", settings.get("pivot_window", 5) * 2))
    current_x = working.index.get_loc(candle.name)
    if not isinstance(current_x, int):
        return []
    min_anchor_age_bars = int(settings.get("chart_trendline_min_anchor_age_bars", settings.get("break_in_trendline_anchor_cluster_bars", settings.get("pivot_window", settings.get("opposing_trendline_pivot_window", 5)))))

    pivot_column = "pivot_low" if direction == "long" else "pivot_high"
    role = "support" if direction == "long" else "resistance"
    pattern = "ascending_support_context" if direction == "long" else "descending_resistance_context"
    anchor_cluster_bars = int(settings.get("break_in_trendline_anchor_cluster_bars", 0))
    min_anchor_move = float(candle["atr"]) * settings.get("break_in_trendline_min_anchor_move_atr", settings.get("secondary_trendline_min_anchor_move_atr", 0.75))
    candidates: list[dict[str, object]] = []

    for lookback in _chart_trendline_lookbacks(settings):
        recent = working if lookback is None else working.iloc[-lookback:]
        pivots = recent[recent[pivot_column].notna()][pivot_column]
        if anchor_cluster_bars > 0:
            pivots = _confirmed_reaction_pivots(working, pivots, anchor_cluster_bars, "low" if direction == "long" else "high")
        if len(pivots) < 2:
            continue
        for first_position in range(len(pivots) - 1):
            for second_position in range(first_position + 1, len(pivots)):
                if second_position != first_position + 1:
                    continue
                first_index = pivots.index[first_position]
                second_index = pivots.index[second_position]
                first_x = working.index.get_loc(first_index)
                second_x = working.index.get_loc(second_index)
                if not isinstance(first_x, int) or not isinstance(second_x, int) or second_x - first_x < min_span:
                    continue
                if current_x - second_x < min_anchor_age_bars:
                    continue
                first_y = float(pivots.iloc[first_position])
                second_y = float(pivots.iloc[second_position])
                if direction == "long" and second_y <= first_y:
                    continue
                if direction == "short" and second_y >= first_y:
                    continue
                if abs(second_y - first_y) < min_anchor_move:
                    continue
                if _trendline_has_middle_pierce(working, first_x, second_x, first_y, second_y, current_x, direction, include_current=False):
                    continue
                if not _trendline_has_later_tap(
                    working,
                    first_x,
                    second_x,
                    first_y,
                    second_y,
                    current_x,
                    direction,
                    settings.get("chart_trendline_tap_tolerance_atr", settings.get("break_in_trendline_atr_tolerance", 0.5)),
                    include_current=True,
                ):
                    continue
                line_value = _project_line_value(first_x, first_y, second_x, second_y, current_x)
                if line_value is None or line_value <= 0:
                    continue
                if direction == "long":
                    distance = float(candle["close"]) - line_value
                else:
                    distance = line_value - float(candle["close"])
                if distance < 0 or distance > max_distance:
                    continue
                cleanliness = _trendline_cleanliness(
                    working,
                    first_x,
                    second_x,
                    first_y,
                    second_y,
                    current_x,
                    direction,
                    settings.get("chart_trendline_tap_tolerance_atr", settings.get("break_in_trendline_atr_tolerance", 0.5)),
                    include_current=False,
                )
                candidates.append(
                    {
                        "line_value": float(line_value),
                        "distance": float(distance),
                        "span_bars": int(second_x - first_x),
                        "patterns": pattern,
                        "first_time": pd.Timestamp(first_index).isoformat(),
                        "second_time": pd.Timestamp(second_index).isoformat(),
                        "first_value": first_y,
                        "second_value": second_y,
                        "lookback": _lookback_label(lookback),
                        "primary": False,
                        "role": role,
                        "cleanliness_score": cleanliness["cleanliness_score"],
                        "close_breaches": cleanliness["close_breaches"],
                        "wick_breaches": cleanliness["wick_breaches"],
                    }
                )
    return sorted(candidates, key=lambda item: (int(item.get("cleanliness_score", 0)), float(item["distance"]), -int(item.get("span_bars", 0))))


def _trendline_key(trendline: dict[str, object]) -> tuple[object, object, object]:
    return (trendline.get("first_time"), trendline.get("second_time"), trendline.get("role") or _trendline_role(trendline))


def _trendline_role(trendline: dict[str, object]) -> str:
    pattern = str(trendline.get("patterns", ""))
    if "resistance" in pattern:
        return "resistance"
    return "support"


def _chart_trendline_lookbacks(settings: dict) -> list[int | None]:
    if "break_in_trendline_lookbacks" in settings or "break_in_trendline_lookback" in settings:
        return _configured_lookbacks(settings, "break_in_trendline_lookbacks", "break_in_trendline_lookback", settings.get("secondary_trendline_lookback", settings.get("corrective_trendline_lookback", 35)))
    if "breakout_lookbacks" in settings or "breakout_lookback" in settings:
        return _configured_lookbacks(settings, "breakout_lookbacks", "breakout_lookback", settings.get("corrective_trendline_lookback", 35))
    return _configured_lookbacks(settings, "corrective_trendline_lookbacks", "corrective_trendline_lookback", 35)


def _trendline_anchor_note(status: dict[str, bool | float | str]) -> str | None:
    first_time = status.get("first_time")
    second_time = status.get("second_time")
    first_value = status.get("first_value")
    second_value = status.get("second_value")
    line_value = status.get("line_value")
    if not isinstance(first_time, str) or not isinstance(second_time, str):
        return None
    if not isinstance(first_value, (int, float)) or not isinstance(second_value, (int, float)) or not isinstance(line_value, (int, float)):
        return None
    return f"Trendline anchors: {first_time[:10]} at {first_value:.4f} to {second_time[:10]} at {second_value:.4f}; projected line now {line_value:.4f}."


def _trendline_context_note(status: dict[str, bool | float | str]) -> str | None:
    trendlines = status.get("trendlines")
    if not isinstance(trendlines, list) or len(trendlines) <= 1:
        return None
    summaries: list[str] = []
    for trendline in trendlines:
        if not isinstance(trendline, dict):
            continue
        first_time = trendline.get("first_time")
        second_time = trendline.get("second_time")
        line_value = trendline.get("line_value")
        if not isinstance(first_time, str) or not isinstance(second_time, str) or not isinstance(line_value, (int, float)):
            continue
        prefix = "primary" if trendline.get("primary") else "context"
        summaries.append(f"{prefix} {first_time[:10]}->{second_time[:10]} now {line_value:.4f}")
    if len(summaries) <= 1:
        return None
    return "Relevant trendlines considered: " + "; ".join(summaries) + "."


def _single_trendline_metadata(status: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in status.items() if key != "trendlines"}


def _breakout_trendline_metadata(status: dict[str, bool | float | str], direction: str) -> dict[str, object] | None:
    first_time = status.get("trendline_first_time")
    second_time = status.get("trendline_second_time")
    first_value = status.get("trendline_first_value")
    second_value = status.get("trendline_second_value")
    if not isinstance(first_time, str) or not isinstance(second_time, str):
        return None
    if not isinstance(first_value, (int, float)) or not isinstance(second_value, (int, float)):
        return None
    breakout_level = status.get("breakout_level")
    metadata: dict[str, object] = {
        "first_time": first_time,
        "second_time": second_time,
        "first_value": float(first_value),
        "second_value": float(second_value),
        "line_value": float(breakout_level) if isinstance(breakout_level, (int, float)) else 0.0,
        "patterns": "descending_resistance_breakout" if direction == "long" else "ascending_support_breakout",
        "primary": True,
        "role": "resistance" if direction == "long" else "support",
    }
    for source_key, target_key in (
        ("trendline_cleanliness_score", "cleanliness_score"),
        ("trendline_close_breaches", "close_breaches"),
        ("trendline_wick_breaches", "wick_breaches"),
    ):
        value = status.get(source_key)
        if isinstance(value, (int, float)):
            metadata[target_key] = int(value)
    return metadata


def _long_term_trendline_context(
    working: pd.DataFrame,
    candle: pd.Series,
    direction: str,
    projected_target: float,
    settings: dict,
) -> dict[str, bool | float | str]:
    entry = float(candle["close"])
    atr_value = float(candle["atr"])
    current_x = working.index.get_loc(candle.name)
    if not isinstance(current_x, int):
        return {"clear": True, "reinforced": False, "reinforcement_note": "", "weakening_note": ""}

    min_span = int(settings.get("long_term_trendline_min_span_bars", 80))
    obstacle_buffer = atr_value * settings.get("long_term_trendline_obstacle_buffer_atr", 0.5)
    reinforcement_distance = atr_value * settings.get("long_term_trendline_reinforcement_max_distance_atr", 4.0)
    lookbacks = _configured_lookbacks(settings, "long_term_trendline_lookbacks", "long_term_trendline_lookback", 260)

    best_reinforcement_distance: float | None = None
    reinforcement_note = ""
    weakening_distance: float | None = None
    weakening_note = ""

    for lookback in lookbacks:
        recent = working if lookback is None else working.iloc[-lookback:]
        reinforcing_column = "pivot_low" if direction == "long" else "pivot_high"
        weakening_column = "pivot_high" if direction == "long" else "pivot_low"

        for line_value, pattern in _projected_trendlines(working, recent, reinforcing_column, current_x, min_span, direction, settings):
            if direction == "long":
                distance = entry - line_value
                reinforces = 0 <= distance <= reinforcement_distance
            else:
                distance = line_value - entry
                reinforces = 0 <= distance <= reinforcement_distance
            if reinforces and (best_reinforcement_distance is None or distance < best_reinforcement_distance):
                best_reinforcement_distance = distance
                reinforcement_note = f"Long-term {pattern} reinforces the breakout context."

        obstacle_role = "resistance" if direction == "long" else "support"
        for line_value, pattern in _projected_obstacle_trendlines(
            working,
            recent,
            weakening_column,
            current_x,
            min_span,
            obstacle_role,
            settings,
        ):
            if direction == "long":
                blocks_path = entry < line_value <= projected_target + obstacle_buffer
                distance = line_value - entry
            else:
                blocks_path = projected_target - obstacle_buffer <= line_value < entry
                distance = entry - line_value
            if blocks_path and (weakening_distance is None or distance < weakening_distance):
                weakening_distance = distance
                weakening_note = f"Long-term {pattern} weakens the setup inside the projected target path."

    if weakening_note:
        return {
            "clear": False,
            "reinforced": best_reinforcement_distance is not None,
            "reinforcement_note": reinforcement_note,
            "weakening_note": weakening_note,
        }
    return {
        "clear": True,
        "reinforced": best_reinforcement_distance is not None,
        "reinforcement_note": reinforcement_note,
        "weakening_note": "",
    }


def _break_in_long_term_stop_line(
    working: pd.DataFrame,
    candle: pd.Series,
    direction: str,
    settings: dict,
) -> dict[str, bool | float | str]:
    current_x = working.index.get_loc(candle.name)
    if not isinstance(current_x, int):
        return {"found": False, "line_value": 0.0, "note": ""}

    min_span = int(settings.get("long_term_trendline_min_span_bars", 80))
    max_distance = float(candle["atr"]) * settings.get("break_in_long_term_stop_max_distance_atr", 4.0)
    entry = float(candle["close"])
    candidates: list[dict[str, object]] = []
    for candidate in _trendline_context_candidates(working, candle, settings, direction, max_distance):
        if int(candidate.get("span_bars", 0)) < min_span:
            continue
        line_value = candidate.get("line_value")
        if not isinstance(line_value, (int, float)):
            continue
        distance = entry - float(line_value) if direction == "long" else float(line_value) - entry
        if distance > 0:
            candidate["long_term"] = True
            candidate["primary"] = False
            candidates.append(candidate)

    if not candidates:
        return {"found": False, "line_value": 0.0, "note": "", "trendline": {}}
    selected = min(candidates, key=lambda candidate: float(candidate["distance"]))
    line_value = float(selected["line_value"])
    role = "support" if direction == "long" else "resistance"
    return {
        "found": True,
        "line_value": line_value,
        "note": f"Nearby long-term {role} trendline at {line_value:.4f} is included in the stop placement.",
        "trendline": selected,
    }


def _breakout_target_path_status(
    working: pd.DataFrame,
    candle: pd.Series,
    direction: str,
    projected_target: float,
    settings: dict,
) -> dict[str, bool | float | str]:
    obstacle = _nearest_target_obstacle(working, candle, direction, projected_target, settings)
    if obstacle is None:
        return {"clear": True, "obstacle": 0.0, "note": ""}
    role = "resistance" if direction == "long" else "support"
    return {
        "clear": False,
        "obstacle": float(obstacle),
        "note": f"Structural {role} at {obstacle:.4f} blocks the Fib target path.",
    }


def _projected_obstacle_trendlines(
    working: pd.DataFrame,
    recent: pd.DataFrame,
    pivot_column: str,
    current_x: int,
    min_span: int,
    role: str,
    settings: dict,
) -> Iterable[tuple[float, str]]:
    pivots = recent[recent[pivot_column].notna()][pivot_column]
    if len(pivots) < 2:
        return
    direction = "long" if role == "support" else "short"
    for first_position in range(len(pivots) - 1):
        second_position = first_position + 1
        first_index = pivots.index[first_position]
        second_index = pivots.index[second_position]
        first_x = working.index.get_loc(first_index)
        second_x = working.index.get_loc(second_index)
        if not isinstance(first_x, int) or not isinstance(second_x, int) or second_x - first_x < min_span:
            continue
        first_y = float(pivots.iloc[first_position])
        second_y = float(pivots.iloc[second_position])
        if _trendline_has_middle_pierce(working, first_x, second_x, first_y, second_y, current_x, direction, include_current=False):
            continue
        if not _trendline_has_later_tap(
            working,
            first_x,
            second_x,
            first_y,
            second_y,
            current_x,
            direction,
            settings.get("long_term_trendline_tap_tolerance_atr", settings.get("break_in_trendline_atr_tolerance", 0.5)),
            include_current=True,
        ):
            continue
        line_value = _project_line_value(first_x, first_y, second_x, second_y, current_x)
        if line_value is not None and line_value > 0:
            yield float(line_value), f"{role} trendline"


def _projected_trendlines(
    working: pd.DataFrame,
    recent: pd.DataFrame,
    pivot_column: str,
    current_x: int,
    min_span: int,
    direction: str,
    settings: dict,
) -> Iterable[tuple[float, str]]:
    pivots = recent[recent[pivot_column].notna()][pivot_column]
    if len(pivots) < 2:
        return
    for first_position in range(len(pivots) - 1):
        for second_position in range(first_position + 1, len(pivots)):
            if second_position != first_position + 1:
                continue
            first_index = pivots.index[first_position]
            second_index = pivots.index[second_position]
            first_x = working.index.get_loc(first_index)
            second_x = working.index.get_loc(second_index)
            if not isinstance(first_x, int) or not isinstance(second_x, int) or second_x - first_x < min_span:
                continue
            first_y = float(pivots.iloc[first_position])
            second_y = float(pivots.iloc[second_position])
            if direction == "long" and second_y <= first_y:
                continue
            if direction == "short" and second_y >= first_y:
                continue
            if _trendline_has_middle_pierce(working, first_x, second_x, first_y, second_y, current_x, direction, include_current=False):
                continue
            if not _trendline_has_later_tap(
                working,
                first_x,
                second_x,
                first_y,
                second_y,
                current_x,
                direction,
                settings.get("long_term_trendline_tap_tolerance_atr", settings.get("break_in_trendline_atr_tolerance", 0.5)),
                include_current=True,
            ):
                continue
            line_value = _project_line_value(first_x, first_y, second_x, second_y, current_x)
            if line_value is None or line_value <= 0:
                continue
            pattern = "ascending support trendline" if direction == "long" else "descending resistance trendline"
            yield float(line_value), pattern


def _breakout_status_for_windows(
    working: pd.DataFrame,
    candle: pd.Series,
    previous: pd.Series,
    direction: str,
    settings: dict,
) -> dict[str, bool | float | str]:
    best_status: dict[str, bool | float | str] | None = None
    best_score = -1
    windows_key = "corrective_trendline_lookbacks" if "corrective_trendline_lookbacks" in settings else "breakout_lookbacks"
    fallback_key = "corrective_trendline_lookback" if "corrective_trendline_lookback" in settings else "breakout_lookback"
    for lookback in _configured_lookbacks(settings, windows_key, fallback_key, 35):
        recent = working if lookback is None else working.iloc[-lookback:]
        status = _breakout_status(recent, working, candle, previous, direction, settings)
        status["lookback"] = _lookback_label(lookback)
        if status["triggered"]:
            return status
        score = _breakout_status_score(status)
        if score > best_score:
            best_status = status
            best_score = score
    if best_status is None:
        return _breakout_status(working.iloc[-settings.get("breakout_lookback", 35):], working, candle, previous, direction, settings)
    return best_status


def _breakout_status_score(status: dict[str, bool | float | str]) -> int:
    keys = (
        "pivot_count",
        "corrective_pivots",
        "candle_conviction",
        "confirmation_window",
        "pivot_break",
        "prior_extreme_clearance",
        "entry_proximity",
        "corrective_line_break",
        "trendline_break",
        "fib_extension",
        "extension_reasonable",
        "triggered",
    )
    return sum(bool(status.get(key)) for key in keys)


def _breakout_status(
    recent: pd.DataFrame,
    working: pd.DataFrame,
    candle: pd.Series,
    previous: pd.Series,
    direction: str,
    settings: dict | None = None,
) -> dict[str, bool | float | str]:
    settings = settings or {}
    status: dict[str, bool | float | str] = {
        "pivot_count": False,
        "corrective_pivots": False,
        "candle_conviction": False,
        "confirmation_window": False,
        "pivot_break": False,
        "prior_extreme_clearance": False,
        "master_candle_patterns": "none",
        "uncertainty_candle": False,
        "uncertainty_patterns": "none",
        "corrective_line_break": False,
        "trendline_break": False,
        "entry_proximity": False,
        "fib_extension": False,
        "extension_distance": 0.0,
        "extension_reasonable": False,
        "used_fallback_target": False,
        "triggered": False,
    }
    pivot_column = "pivot_high" if direction == "long" else "pivot_low"
    price_column = "low" if direction == "long" else "high"
    pivots = recent[recent[pivot_column].notna()][pivot_column]
    if len(pivots) < 2:
        return status
    status["pivot_count"] = True

    best_status = status
    best_score = _breakout_status_score(status)
    min_span = int(settings.get("breakout_trendline_min_span_bars", settings.get("pivot_window", 5) * 2))
    recent_anchor_lookback = int(settings.get("corrective_recent_pivot_lookback", settings.get("breakout_recent_pivot_lookback", settings.get("breakout_lookback", 35))))
    current_x = working.index.get_loc(candle.name)
    if not isinstance(current_x, int):
        return status
    pivot_tolerance = float(candle["atr"]) * 0.25
    for first_position in range(len(pivots) - 1):
        for second_position in range(first_position + 1, len(pivots)):
            if second_position != first_position + 1:
                continue
            if second_position != len(pivots) - 1:
                continue
            first_index = pivots.index[first_position]
            second_index = pivots.index[second_position]
            first_x = working.index.get_loc(first_index)
            second_x = working.index.get_loc(second_index)
            if not isinstance(first_x, int) or not isinstance(second_x, int) or second_x - first_x < min_span:
                continue
            if current_x - second_x > recent_anchor_lookback:
                continue
            first_y = float(pivots.iloc[first_position])
            second_y = float(pivots.iloc[second_position])
            if direction == "long" and second_y > first_y + pivot_tolerance:
                continue
            if direction == "short" and second_y < first_y - pivot_tolerance:
                continue
            if not _trendline_has_later_tap(
                working,
                first_x,
                second_x,
                first_y,
                second_y,
                current_x,
                "short" if direction == "long" else "long",
                settings.get("corrective_trendline_tap_tolerance_atr", settings.get("breakout_close_tolerance_atr", 0.5)),
                include_current=False,
            ):
                continue
            candidate = _breakout_pair_status(
                status,
                recent,
                working,
                candle,
                previous,
                direction,
                settings,
                first_x,
                first_y,
                second_x,
                second_y,
                price_column,
            )
            if candidate["triggered"]:
                return candidate
            score = _breakout_status_score(candidate)
            if score > best_score:
                best_status = candidate
                best_score = score
    return best_status


def _breakout_pair_status(
    base_status: dict[str, bool | float | str],
    recent: pd.DataFrame,
    working: pd.DataFrame,
    candle: pd.Series,
    previous: pd.Series,
    direction: str,
    settings: dict,
    first_x: int,
    first_y: float,
    second_x: int,
    second_y: float,
    price_column: str,
) -> dict[str, bool | float | str]:
    status = dict(base_status)
    status["corrective_pivots"] = True
    status["trendline_first_time"] = pd.Timestamp(working.index[first_x]).isoformat()
    status["trendline_second_time"] = pd.Timestamp(working.index[second_x]).isoformat()
    status["trendline_first_value"] = first_y
    status["trendline_second_value"] = second_y

    current_x = working.index.get_loc(candle.name)
    previous_x = working.index.get_loc(previous.name)
    if not isinstance(current_x, int) or not isinstance(previous_x, int):
        return status

    current_line = _project_line_value(first_x, first_y, second_x, second_y, current_x)
    previous_line = _project_line_value(first_x, first_y, second_x, second_y, previous_x)
    if current_line is None or previous_line is None:
        return status
    cleanliness = _trendline_cleanliness(
        working,
        first_x,
        second_x,
        first_y,
        second_y,
        current_x,
        "short" if direction == "long" else "long",
        settings.get("corrective_trendline_tap_tolerance_atr", settings.get("breakout_close_tolerance_atr", 0.5)),
        include_current=False,
    )
    status["trendline_cleanliness_score"] = cleanliness["cleanliness_score"]
    status["trendline_close_breaches"] = cleanliness["close_breaches"]
    status["trendline_wick_breaches"] = cleanliness["wick_breaches"]

    close_tolerance = float(candle["atr"]) * settings.get("breakout_close_tolerance_atr", 0.5)
    max_entry_distance = float(candle["atr"]) * settings.get("breakout_max_entry_distance_atr", 0.75)
    recent_breakout_window = recent.tail(settings.get("breakout_window_bars", 5))
    first_signal_window_x = working.index.get_loc(recent_breakout_window.index[0])
    if not isinstance(first_signal_window_x, int):
        return status
    if _trendline_has_middle_pierce(
        working,
        first_x,
        second_x,
        first_y,
        second_y,
        first_signal_window_x,
        "short" if direction == "long" else "long",
        include_current=False,
    ):
        return status
    prior_structure = recent.loc[: recent_breakout_window.index[0]].iloc[:-1]
    if prior_structure.empty:
        prior_structure = recent.iloc[:-len(recent_breakout_window)]
    master_candle = _recent_master_candle_status(working, recent_breakout_window, direction, settings)
    status["candle_conviction"] = bool(master_candle["passed"])
    status["confirmation_window"] = bool(master_candle["passed"])
    status["master_candle_patterns"] = str(master_candle["patterns"])
    status["master_candle_time"] = str(master_candle.get("candle_time", ""))
    status["master_candle_open"] = master_candle.get("open")
    status["master_candle_close"] = master_candle.get("close")
    status["uncertainty_candle"] = master_candle["uncertainty_patterns"] != "none"
    status["uncertainty_patterns"] = str(master_candle["uncertainty_patterns"])
    if not master_candle["passed"]:
        return status

    if direction == "long":
        recent_pierce = False
        recent_pivot_break = False
        prior_extreme = float(prior_structure["high"].max()) if not prior_structure.empty else second_y
        status["prior_extreme"] = prior_extreme
        for index, row in recent_breakout_window.iterrows():
            x = working.index.get_loc(index)
            if not isinstance(x, int):
                continue
            line_value = _project_line_value(first_x, first_y, second_x, second_y, x)
            if line_value is None:
                continue
            if float(row["high"]) > line_value:
                recent_pierce = True
            position = working.index.get_loc(index)
            prior_close = working.iloc[position - 1]["close"] if isinstance(position, int) and position > 0 else row["close"]
            if row["close"] > second_y and float(row["high"]) > second_y and (row["close"] > row["open"] or row["close"] > prior_close):
                recent_pivot_break = True
            if row["close"] > prior_extreme and float(row["high"]) > prior_extreme:
                status["prior_extreme_clearance"] = True
        breakout_level = current_line
        entry_proximity = 0 <= float(candle["close"] - breakout_level) <= max_entry_distance
        corrective_line_break = (
            recent_pierce
            and recent_pivot_break
            and candle["close"] >= current_line - close_tolerance
            and float(candle["close"]) > current_line
        )
        trendline_break = corrective_line_break and bool(status["prior_extreme_clearance"])
    else:
        recent_pierce = False
        recent_pivot_break = False
        prior_extreme = float(prior_structure["low"].min()) if not prior_structure.empty else second_y
        status["prior_extreme"] = prior_extreme
        for index, row in recent_breakout_window.iterrows():
            x = working.index.get_loc(index)
            if not isinstance(x, int):
                continue
            line_value = _project_line_value(first_x, first_y, second_x, second_y, x)
            if line_value is None:
                continue
            if float(row["low"]) < line_value:
                recent_pierce = True
            position = working.index.get_loc(index)
            prior_close = working.iloc[position - 1]["close"] if isinstance(position, int) and position > 0 else row["close"]
            if row["close"] < second_y and float(row["low"]) < second_y and (row["close"] < row["open"] or row["close"] < prior_close):
                recent_pivot_break = True
            if row["close"] < prior_extreme and float(row["low"]) < prior_extreme:
                status["prior_extreme_clearance"] = True
        breakout_level = current_line
        entry_proximity = 0 <= float(breakout_level - candle["close"]) <= max_entry_distance
        corrective_line_break = (
            recent_pierce
            and recent_pivot_break
            and candle["close"] <= current_line + close_tolerance
            and float(candle["close"]) < current_line
        )
        trendline_break = corrective_line_break and bool(status["prior_extreme_clearance"])
    status["pivot_break"] = bool(recent_pivot_break)
    status["entry_proximity"] = bool(entry_proximity)
    status["corrective_line_break"] = bool(corrective_line_break)
    if not entry_proximity:
        return status

    status["trendline_break"] = bool(trendline_break)
    if not trendline_break:
        return status

    fib_extension = _trend_based_fib_extension(working, prior_structure, first_signal_window_x, direction, settings)
    if fib_extension is None:
        return status
    status.update(fib_extension)
    status["fib_extension"] = True

    extension_distance = _breakout_extension_distance(recent, working, first_x, first_y, second_x, second_y, price_column)
    status["extension_distance"] = extension_distance
    fallback_target_atr = settings.get("breakout_fallback_target_atr", 1.5)
    allow_fallback_target = bool(settings.get("breakout_allow_fallback_target", True))
    if extension_distance <= 0:
        if not allow_fallback_target:
            return status
        status["extension_distance"] = float(candle["atr"]) * fallback_target_atr
        status["extension_reasonable"] = True
        status["used_fallback_target"] = True
        status["triggered"] = True
        return status
    if extension_distance > float(candle["atr"]) * settings.get("breakout_max_extension_atr", 6.0):
        if not allow_fallback_target:
            return status
        status["extension_distance"] = float(candle["atr"]) * fallback_target_atr
        status["extension_reasonable"] = True
        status["used_fallback_target"] = True
        status["triggered"] = True
        return status

    status["extension_reasonable"] = True
    status["triggered"] = True
    return status


def _trend_based_fib_extension(
    working: pd.DataFrame,
    prior_structure: pd.DataFrame,
    first_signal_window_x: int,
    direction: str,
    settings: dict,
) -> dict[str, float | str] | None:
    if prior_structure.empty:
        return None

    impulse_extreme_column = "high" if direction == "long" else "low"
    retracement_pivot_column = "pivot_low" if direction == "long" else "pivot_high"
    impulse_extreme_index = prior_structure[impulse_extreme_column].idxmax() if direction == "long" else prior_structure[impulse_extreme_column].idxmin()
    impulse_extreme_x = working.index.get_loc(impulse_extreme_index)
    if not isinstance(impulse_extreme_x, int):
        return None

    origin_pivots = working.iloc[:impulse_extreme_x][retracement_pivot_column].dropna()
    retracement_pivots = working.iloc[impulse_extreme_x + 1 : first_signal_window_x][retracement_pivot_column].dropna()
    if origin_pivots.empty or retracement_pivots.empty:
        return None

    origin_index = origin_pivots.index[-1]
    retracement_index = retracement_pivots.idxmin() if direction == "long" else retracement_pivots.idxmax()
    origin = float(origin_pivots.iloc[-1])
    impulse_extreme = next(
        float(value)
        for index, value in prior_structure[impulse_extreme_column].items()
        if index == impulse_extreme_index
    )
    retracement = float(retracement_pivots.loc[retracement_index])
    impulse_range = impulse_extreme - origin
    if (direction == "long" and impulse_range <= 0) or (direction == "short" and impulse_range >= 0):
        return None

    stop_level = float(settings.get("breakout_fib_stop_level", 0.0))
    target_1_level = float(settings.get("breakout_fib_target_1_level", 1.0))
    target_2_level = float(settings.get("breakout_fib_target_2_level", 1.618))
    stop = retracement + impulse_range * stop_level
    target_1 = retracement + impulse_range * target_1_level
    target_2 = retracement + impulse_range * target_2_level
    if direction == "long" and not (stop < target_1 < target_2):
        return None
    if direction == "short" and not (stop > target_1 > target_2):
        return None

    return {
        "fib_origin_time": pd.Timestamp(origin_index).isoformat(),
        "fib_impulse_time": pd.Timestamp(impulse_extreme_index).isoformat(),
        "fib_retracement_time": pd.Timestamp(retracement_index).isoformat(),
        "fib_origin": origin,
        "fib_impulse": impulse_extreme,
        "fib_retracement": retracement,
        "fib_stop": float(stop),
        "fib_target_1": float(target_1),
        "fib_target_2": float(target_2),
        "fib_stop_level": stop_level,
        "fib_target_1_level": target_1_level,
        "fib_target_2_level": target_2_level,
    }


def _breakout_fib_levels(status: dict[str, bool | float | str]) -> dict[str, float | str] | None:
    required = ("fib_stop", "fib_target_1", "fib_target_2", "fib_origin", "fib_impulse", "fib_retracement")
    if not all(isinstance(status.get(key), (int, float)) for key in required):
        return None
    return {
        "origin_time": str(status.get("fib_origin_time", "")),
        "impulse_time": str(status.get("fib_impulse_time", "")),
        "retracement_time": str(status.get("fib_retracement_time", "")),
        "origin": float(status["fib_origin"]),
        "impulse": float(status["fib_impulse"]),
        "retracement": float(status["fib_retracement"]),
        "stop": float(status["fib_stop"]),
        "target_1": float(status["fib_target_1"]),
        "target_2": float(status["fib_target_2"]),
        "stop_level": float(status.get("fib_stop_level", 0.0)),
        "target_1_level": float(status.get("fib_target_1_level", 1.0)),
        "target_2_level": float(status.get("fib_target_2_level", 1.618)),
    }


def _fib_anchor_note(fib_levels: dict[str, float | str]) -> str | None:
    origin = fib_levels.get("origin")
    impulse = fib_levels.get("impulse")
    retracement = fib_levels.get("retracement")
    if not isinstance(origin, (int, float)) or not isinstance(impulse, (int, float)) or not isinstance(retracement, (int, float)):
        return None
    return f"Trend-Based Fib Extension anchors: A {origin:.4f}, B {impulse:.4f}, C {retracement:.4f}."


def _breakout_extension_distance(
    recent: pd.DataFrame,
    working: pd.DataFrame,
    first_x: int,
    first_y: float,
    second_x: int,
    second_y: float,
    price_column: str,
) -> float:
    max_distance = 0.0
    for index, row in recent.iterrows():
        x = working.index.get_loc(index)
        if not isinstance(x, int):
            continue
        line_value = _project_line_value(first_x, first_y, second_x, second_y, x)
        if line_value is None:
            continue
        if price_column == "low":
            distance = line_value - float(row[price_column])
        else:
            distance = float(row[price_column]) - line_value
        max_distance = max(max_distance, distance)
    return max_distance


def _structure_capped_extension_distance(
    working: pd.DataFrame,
    candle: pd.Series,
    direction: str,
    extension_distance: float,
    settings: dict,
) -> float:
    if extension_distance <= 0:
        return 0.0
    entry = float(candle["close"])
    target = entry + extension_distance if direction == "long" else entry - extension_distance
    buffer_distance = float(candle["atr"]) * settings.get("target_obstacle_buffer_atr", 0.25)
    obstacle = _nearest_target_obstacle(working, candle, direction, target, settings)
    if obstacle is None:
        return extension_distance
    if direction == "long":
        capped_target = obstacle - buffer_distance
        return max(0.0, capped_target - entry)
    capped_target = obstacle + buffer_distance
    return max(0.0, entry - capped_target)


def _breakout_trade_plan(
    candle: pd.Series,
    settings: dict,
    direction: str,
    entry: float,
    raw_stop: float,
    target_1: float,
    target_2: float,
    breakout: dict[str, bool | float | str],
) -> TradePlan | None:
    plan = _trade_plan(candle, settings, direction, entry, raw_stop, target_1, target_2)
    if plan is None or plan["rr"] < settings["minimum_rr"]:
        return None

    plan["target_note"] = "Stop, Target 1, and final target use TradingView Trend-Based Fib Extension levels 0.0, 1.0, and 1.618."
    return plan


def _breakout_target_candidates(working: pd.DataFrame, candle: pd.Series, settings: dict, direction: str, projected_target: float) -> list[tuple[float, str]]:
    entry = float(candle["close"])
    buffer_distance = float(candle["atr"]) * settings.get("target_obstacle_buffer_atr", 0.25)
    levels = _target_obstacle_levels(working, candle, direction, settings)
    if direction == "long":
        targets = [level - buffer_distance for level in levels if entry < level <= projected_target and level - buffer_distance > entry]
        return [(target, "Target uses the nearest structural resistance that still preserves configured reward/risk.") for target in sorted(set(targets))]

    targets = [level + buffer_distance for level in levels if projected_target <= level < entry and level + buffer_distance < entry]
    return [(target, "Target uses the nearest structural support that still preserves configured reward/risk.") for target in sorted(set(targets), reverse=True)]


def _unique_target_candidates(candidates: list[tuple[float, str]], direction: str) -> list[tuple[float, str]]:
    ordered = sorted(candidates, key=lambda item: item[0], reverse=direction == "short")
    unique: list[tuple[float, str]] = []
    seen: set[float] = set()
    for target, note in ordered:
        rounded = round(float(target), 8)
        if rounded in seen:
            continue
        seen.add(rounded)
        unique.append((float(target), note))
    return unique


def _corrective_break_in_target(
    working: pd.DataFrame,
    candle: pd.Series,
    direction: str,
    corrective_status: dict[str, bool | float | str],
    settings: dict,
) -> float:
    entry = float(candle["close"])
    prior_extreme = float(corrective_status.get("prior_extreme", 0.0)) if isinstance(corrective_status.get("prior_extreme", 0.0), (int, float)) else 0.0
    if direction == "long":
        if prior_extreme > entry:
            return prior_extreme
        structural_extension = _nearest_structural_target_extension_distance(working, candle, direction, settings)
        if structural_extension > 0:
            return entry + structural_extension
        resistance = float(candle.get("resistance", 0.0))
        return resistance if resistance > entry else 0.0

    if prior_extreme > 0 and prior_extreme < entry:
        return prior_extreme
    structural_extension = _nearest_structural_target_extension_distance(working, candle, direction, settings)
    if structural_extension > 0:
        return entry - structural_extension
    support = float(candle.get("support", 0.0))
    return support if 0 < support < entry else 0.0


def _break_in_trade_plan(working: pd.DataFrame, candle: pd.Series, settings: dict, direction: str, entry: float, raw_stop: float) -> TradePlan | None:
    for raw_target, target_note in _break_in_target_candidates(working, candle, settings, direction):
        plan = _trade_plan(candle, settings, direction, entry, raw_stop, raw_target)
        if plan is not None and plan["rr"] >= settings["minimum_rr"]:
            plan["target_note"] = target_note
            return plan
    return None


def _break_in_target_candidates(working: pd.DataFrame, candle: pd.Series, settings: dict, direction: str) -> list[tuple[float, str]]:
    entry = float(candle["close"])
    levels: list[float] = []
    if direction == "long":
        levels.extend(_structural_pivot_levels(working, settings, "pivot_high"))
        levels.extend(_finite_candle_levels(candle, "resistance"))
        levels.extend(_finite_candle_levels(candle, "long_term_resistance"))
        candidates = [level for level in levels if level > entry]
        fallback = float(candle.get("resistance", 0.0))
        note = "Target uses the first structural resistance obstacle, matching the Excel break-in rule."
        fallback_note = "Target uses the first rolling resistance obstacle."
        ordered = sorted(set(candidates))
        if fallback > entry:
            ordered.append(fallback)
        return [(target, note if target != fallback else fallback_note) for target in _unique_ordered(ordered)]

    levels.extend(_structural_pivot_levels(working, settings, "pivot_low"))
    levels.extend(_finite_candle_levels(candle, "support"))
    levels.extend(_finite_candle_levels(candle, "long_term_support"))
    candidates = [level for level in levels if 0 < level < entry]
    fallback = float(candle.get("support", 0.0))
    note = "Target uses the first structural support obstacle, matching the Excel break-in rule."
    fallback_note = "Target uses the first rolling support obstacle."
    ordered = sorted(set(candidates), reverse=True)
    if 0 < fallback < entry:
        ordered.append(fallback)
    return [(target, note if target != fallback else fallback_note) for target in _unique_ordered(ordered)]


def _structural_pivot_levels(working: pd.DataFrame, settings: dict, pivot_column: str) -> list[float]:
    levels: list[float] = []
    for lookback in _configured_lookbacks(settings, "break_in_target_lookbacks", "break_in_target_lookback", settings.get("target_obstacle_lookback", 260)):
        recent = working if lookback is None else working.iloc[-lookback:]
        if pivot_column in recent:
            levels.extend(float(value) for value in recent[recent[pivot_column].notna()][pivot_column].tolist())
    return levels


def _unique_ordered(values: list[float]) -> list[float]:
    unique: list[float] = []
    seen: set[float] = set()
    for value in values:
        rounded = round(float(value), 8)
        if rounded in seen:
            continue
        seen.add(rounded)
        unique.append(float(value))
    return unique


def _nearest_structural_target_extension_distance(
    working: pd.DataFrame,
    candle: pd.Series,
    direction: str,
    settings: dict,
) -> float:
    entry = float(candle["close"])
    buffer_distance = float(candle["atr"]) * settings.get("target_obstacle_buffer_atr", 0.25)
    target_bound = float("inf") if direction == "long" else float("-inf")
    obstacle = _nearest_target_obstacle(working, candle, direction, target_bound, settings)
    if obstacle is None:
        return 0.0
    if direction == "long":
        return max(0.0, obstacle - buffer_distance - entry)
    return max(0.0, entry - obstacle - buffer_distance)


def _nearest_target_obstacle(
    working: pd.DataFrame,
    candle: pd.Series,
    direction: str,
    target: float,
    settings: dict,
) -> float | None:
    entry = float(candle["close"])
    levels = _target_obstacle_levels(working, candle, direction, settings)
    if direction == "long":
        obstacles = [level for level in levels if entry < level < target]
        return min(obstacles) if obstacles else None

    obstacles = [level for level in levels if target < level < entry]
    return max(obstacles) if obstacles else None


def _target_obstacle_levels(working: pd.DataFrame, candle: pd.Series, direction: str, settings: dict) -> list[float]:
    levels: list[float] = []
    if direction == "long":
        pivot_column = "pivot_high"
        levels.extend(_finite_candle_levels(candle, "long_term_resistance"))
        levels.extend(_finite_candle_levels(candle, "resistance"))
    else:
        pivot_column = "pivot_low"
        levels.extend(_finite_candle_levels(candle, "long_term_support"))
        levels.extend(_finite_candle_levels(candle, "support"))
    for lookback in _configured_lookbacks(settings, "target_obstacle_lookbacks", "target_obstacle_lookback", 260):
        recent = working if lookback is None else working.iloc[-lookback:]
        levels.extend(float(value) for value in recent[recent[pivot_column].notna()][pivot_column].tolist())
    return _unique_ordered(levels)


def _project_line_value(first_x: int, first_y: float, second_x: int, second_y: float, x: int) -> float | None:
    if second_x == first_x:
        return None
    slope_value = (second_y - first_y) / (second_x - first_x)
    return first_y + slope_value * (x - first_x)


def _reward_risk(entry: float, stop: float, target: float, direction: str) -> float:
    risk = abs(entry - stop)
    if risk == 0:
        return 0.0
    reward = target - entry if direction == "long" else entry - target
    return reward / risk


def _trade_plan(candle: pd.Series, settings: dict, direction: str, entry: float, raw_stop: float, raw_target: float, raw_target_2: float | None = None) -> TradePlan | None:
    stop = float(raw_stop)
    target_1 = float(raw_target)
    target_2 = float(raw_target if raw_target_2 is None else raw_target_2)

    if direction == "long":
        stop_distance = entry - stop
        target_1_distance = target_1 - entry
        target_2_distance = target_2 - entry
        if stop_distance <= 0 or target_1_distance <= 0 or target_2_distance <= 0:
            return None
    else:
        stop_distance = stop - entry
        target_1_distance = entry - target_1
        target_2_distance = entry - target_2
        if stop_distance <= 0 or target_1_distance <= 0 or target_2_distance <= 0:
            return None

    risk_pct = stop_distance / entry
    target_1_pct = target_1_distance / entry
    target_2_pct = target_2_distance / entry
    if risk_pct <= 0:
        return None
    rr = ((target_1_pct + target_2_pct) / 2) / risk_pct
    return {
        "stop": float(stop),
        "target": float(target_2),
        "target_1": float(target_1),
        "target_2": float(target_2),
        "checkpoint": float(target_1),
        "rr": float(rr),
    }


def _excel_risk_pct_for_rr(rr: float) -> float:
    if rr < 1.5:
        return 0.0
    if rr <= 5.0:
        return 0.02
    return 0.03


def _position_size(account_value: float, risk_pct: float, position_value_pct: float, entry: float, stop: float) -> float | None:
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0 or entry <= 0:
        return None
    risk_budget = account_value * risk_pct
    risk_based_size = risk_budget / risk_per_unit
    position_value_budget = account_value * position_value_pct
    value_based_size = position_value_budget / entry
    return min(risk_based_size, value_based_size)


def _build_signal(
    pattern: str,
    direction: str,
    timeframe: str,
    instrument: Instrument,
    candle: pd.Series,
    entry: float,
    stop: float,
    target: float,
    rr: float,
    account_value: float,
    risk_pct: float,
    position_value_pct: float,
    notes: list[str],
    metadata: dict | None = None,
) -> TradeSignal:
    candle_time = pd.Timestamp(str(candle.name)).isoformat()
    selected_risk_pct = _excel_risk_pct_for_rr(rr)
    signal_metadata = dict(metadata or {})
    signal_metadata.setdefault(
        "master_candle",
        {
            "time": candle_time,
            "open": float(candle["open"]),
            "close": float(candle["close"]),
        },
    )
    management = signal_metadata.get("management")
    if isinstance(management, dict):
        management.setdefault("risk_allocation_pct", selected_risk_pct)
        management.setdefault("risk_model", "Excel R/R table")
    return TradeSignal(
        pattern,
        direction,
        timeframe,
        instrument,
        candle_time,
        entry,
        stop,
        target,
        rr,
        _position_size(account_value, selected_risk_pct, position_value_pct, entry, stop),
        notes,
        signal_metadata,
    )