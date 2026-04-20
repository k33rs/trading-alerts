# Strategy Specification

This document describes the technical-analysis logic currently implemented in the bot. It is a code-level strategy spec, not a discretionary chart-reading guide.

The implementation lives primarily in `src/trading_bot/patterns.py` and uses the indicators in `src/trading_bot/indicators.py`.

## Strategy Overview

The bot evaluates three pattern families:

- Break-in
- Break-out
- Bollinger reversion

The first two are trend-following structures on higher timeframes. The third is a mean-reversion structure on the 1-hour chart.

The strategy is designed as a systematic approximation of the chart concepts from the notes. It does not attempt to reproduce discretionary trendline drawing or subjective support/resistance marking exactly.

## Shared Concepts

A few building blocks are reused across the patterns:

- Trend filter: EMA alignment with 20-period and 50-period EMAs.
- Volatility filter: ATR.
- Structure levels: short-term secondary/corrective pivot trendlines for break-in and break-out, filtered by the main trend direction.
- Momentum proxy: close-to-close difference over a configurable lookback.
- Position sizing: size is based on live IBKR NetLiquidation and capped by both risk-per-trade percentage and position-value percentage from config.

Indicators are implemented as follows:

- EMA: exponential moving average.
- ATR: rolling average true range.
- Average volume: rolling mean of volume.
- Bollinger Bands: rolling mean plus/minus standard deviation multiples.

## Break-in

Break-in is the pullback-into-structure pattern.

Configured timeframes:

- `4h`
- `1d`
- `1wk`
- `1mo`

Current config values:

- `ema_fast: 20`
- `ema_slow: 50`
- `ema_long: 100`
- `trend_slope_lookback: 3`
- `trend_slope_min_atr: 0.08`
- `long_term_trend_slope_lookback: 10`
- `long_term_trend_slope_min_atr: 0.02`
- `secular_trend_lookbacks: [260, 520, 1040, all]`
- `secular_trend_min_score: 2.0`
- `support_resistance_lookback: 20`
- `support_resistance_lookbacks: [20, 55, 120, 220, all]`
- `support_resistance_atr_tolerance: 1.5`
- `break_in_level_test_window_bars: 5`
- `break_in_trendline_lookbacks: [35, 70, 140, 260, all]`
- `break_in_trendline_min_span_bars: 10`
- `break_in_trendline_anchor_cluster_bars: 8`
- Trendline anchors must be consecutive confirmed swing highs or swing lows after clustering, not neighboring candles. Lines that skip an intervening same-side swing are rejected. Any middle candle that pierces through the candidate trendline invalidates it. Direct break-in signals may use the signal candle as the third tap; breakout signal-window pierces are allowed only as the breakout event itself. Non-signal context and breakout trendlines require a later tap within ATR tolerance.
- `break_in_trendline_max_context_lines: 4`
- `break_in_chart_trendline_max_lines_per_side: 3`
- `break_in_chart_trendline_max_distance_atr: 8.0`
- `chart_horizontal_level_max_distance_atr: 12.0`
- `chart_horizontal_level_max_per_side: 4`
- `chart_trendline_min_anchor_age_bars: 8`
- `breakout_chart_trendline_max_lines_per_side: 3`
- `breakout_chart_trendline_max_distance_atr: 8.0`
- `break_in_trendline_min_anchor_move_atr: 0.75`
- `break_in_trendline_atr_tolerance: 0.5`
- `break_in_target_obstacle_buffer_atr: 0.25`
- `break_in_allow_corrective_reclaim: false`
- `long_term_level_lookback: 220`
- `long_term_level_lookbacks: [55, 120, 220, all]`
- `long_term_level_atr_tolerance: 0.75`
- `opposing_trendline_lookback: 220`
- `opposing_trendline_lookbacks: [120, 220, all]`
- `opposing_trendline_pivot_window: 5`
- `opposing_trendline_min_span_bars: 40`
- `opposing_trendline_atr_tolerance: 0.75`
- `long_term_trendline_lookbacks: [120, 260, all]`
- `long_term_trendline_min_span_bars: 80`
- `break_in_long_term_stop_max_distance_atr: 4.0`
- `corrective_trendline_lookbacks: [35, 70, 140, 260, all]`
- `corrective_recent_pivot_lookback: 35`
- `master_candle_min_body_to_range: 0.15`
- `master_candle_strong_body_to_range: 0.55`
- `master_candle_close_zone: 0.25`
- `master_candle_uncertainty_body_max: 0.12`
- `master_candle_spinning_top_body_max: 0.25`
- `master_candle_uncertainty_wick_min: 0.30`
- `volume_ratio_min: 1.2`
- `target_obstacle_lookbacks: [120, 260, all]`
- `minimum_rr: 2.0`

### Long setup

The bot looks for:

- bullish trend: fast EMA above slow EMA and a fast-EMA slope greater than `0.08 ATR` over the last `3` bars
- long-term trend: either close and slow EMA above the `100` EMA with the `100` EMA slope greater than `0.02 ATR` over the last `10` bars, or a decisive bullish multi-year secular trend across the configured `260`, `520`, `1040`, and full-history windows
- no nearby long-term rolling resistance level within `0.75 ATR`, checked across medium, long, and full-history windows
- price touches or slightly pierces an ascending support trendline built from pivot lows, then closes back upward in the direction of the main positive trend; the second trendline anchor must be the latest pivot low in the selected window, and the anchor-to-anchor rise must be meaningful relative to ATR
- horizontal support/resistance levels remain context for clearance, stops, targets, and obstacles, but they are no longer the primary break-in trigger
- if the same candle also breaks through a descending corrective resistance trendline built from pivot highs, it can qualify separately as a break-out
- a bullish master candle near support; textbook candle patterns reinforce the setup when present but are not mandatory
- volume at least `1.2x` average volume strengthens the setup but is not mandatory

Recognized bullish candle patterns include:

- hammer-like candle with a long lower wick
- bullish pin bar
- bullish engulfing-style candle
- piercing-line-style candle
- strong bullish marubozu-style candle

Every master candle must close in the signal direction and in the configured directional close zone. Doji, long-legged doji, and spinning-top-like candles are reported as uncertainty context but can never qualify a directional setup by themselves.

Trade construction:

- entry: latest close
- stop: lower of candle low, the bounce trendline, and a validated ascending long-term support trendline within `4.0 ATR`, matching the Excel break-in rule's structural-support intent
- target 1 and final target: first resistance obstacle; when only one obstacle is available, Target 1 and Target 2 are the same level
- management: checkpoint is Target 1; reward/risk uses the Excel formula, average of Target 1 % and Target 2 % divided by stop %
- minimum reward-to-risk: `2.0`

### Short setup

The logic is mirrored:

- bearish trend: fast EMA below slow EMA and a fast-EMA slope below `-0.08 ATR` over the last `3` bars
- long-term trend: either close and slow EMA below the `100` EMA with the `100` EMA slope below `-0.02 ATR` over the last `10` bars, or a decisive bearish multi-year secular trend across the configured `260`, `520`, `1040`, and full-history windows
- no nearby long-term rolling support level within `0.75 ATR`, checked across medium, long, and full-history windows
- price touches or slightly pierces a descending resistance trendline built from pivot highs, then closes back downward in the direction of the main negative trend; the second trendline anchor must be the latest pivot high in the selected window, and the anchor-to-anchor decline must be meaningful relative to ATR
- horizontal support/resistance levels remain context for clearance, stops, targets, and obstacles, but they are no longer the primary break-in trigger
- if the same candle also breaks through an ascending corrective support trendline built from pivot lows, it can qualify separately as a break-out
- a bearish master candle near resistance; textbook candle patterns reinforce the setup when present but are not mandatory
- volume at least `1.2x` average volume strengthens the setup but is not mandatory

Recognized bearish candle patterns include:

- shooting-star-like candle with a long upper wick
- bearish pin bar
- bearish engulfing-style candle
- dark-cloud-cover-style candle
- strong bearish marubozu-style candle

Trade construction:

- entry: latest close
- stop: higher of candle high, the rejection trendline, and a validated descending long-term resistance trendline within `4.0 ATR`, matching the Excel break-in rule's structural-resistance intent
- target 1 and final target: first support obstacle; when only one obstacle is available, Target 1 and Target 2 are the same level
- management: checkpoint is Target 1; reward/risk uses the Excel formula, average of Target 1 % and Target 2 % divided by stop %
- minimum reward-to-risk: `2.0`

## Break-out

Break-out is the trend continuation pattern after a pullback or pause. It can fire from either a horizontal support/resistance break or a corrective trendline break, as long as the signal also passes trend, candle, proximity, and reward/risk checks.

Configured timeframes:

- `4h`
- `1d`
- `1wk`

Current config values:

- `ema_fast: 20`
- `ema_slow: 50`
- `ema_long: 100`
- `breakout_lookback: 35`
- `breakout_lookbacks: [35, 70, 140, 260, all]`
- `breakout_recent_pivot_lookback: 35`
- `pivot_window: 5`
- `atr_period: 20`
- `trend_slope_lookback: 3`
- `long_term_trend_slope_lookback: 10`
- `long_term_trend_slope_min_atr: 0.02`
- `secular_trend_lookbacks: [260, 520, 1040, all]`
- `secular_trend_min_score: 2.0`
- `long_term_level_lookback: 220`
- `long_term_level_lookbacks: [55, 120, 220, all]`
- `long_term_level_atr_tolerance: 0.75`
- `long_term_trendline_lookbacks: [120, 260, all]`
- `long_term_trendline_min_span_bars: 80`
- `long_term_trendline_obstacle_buffer_atr: 0.5`
- `long_term_trendline_reinforcement_max_distance_atr: 4.0`
- `long_term_trendline_chart_max_distance_atr: 12.0`
- `trend_mode: aligned_or_emerging`
- `breakout_window_bars: 5`
- `breakout_master_candle_window_bars: 3`
- `breakout_signal_valid_window_bars: 3`
- `breakout_fib_stop_level: 0.0`
- `breakout_fib_target_1_level: 1.0`
- `breakout_fib_target_2_level: 1.618`
- `breakout_close_tolerance_atr: 0.5`
- `breakout_max_entry_distance_atr: 0.75`
- `breakout_max_extension_atr: 6.0`
- `master_candle_min_body_to_range: 0.15`
- `master_candle_strong_body_to_range: 0.55`
- `master_candle_close_zone: 0.25`
- `master_candle_uncertainty_body_max: 0.12`
- `master_candle_spinning_top_body_max: 0.25`
- `master_candle_uncertainty_wick_min: 0.30`
- `breakout_allow_fallback_target: false`
- `breakout_fallback_target_atr: 1.5`
- `target_obstacle_lookbacks: [120, 260, all]`
- `target_obstacle_buffer_atr: 0.25`
- `chart_horizontal_level_max_distance_atr: 12.0`
- `chart_horizontal_level_max_per_side: 4`
- `minimum_rr: 1.5`

### Long setup

The bot looks for:

- bullish trend: fast EMA above slow EMA, or an emerging bullish breakout where price has reclaimed the fast EMA with positive fast-EMA slope
- long-term trend: either close and slow EMA above the `100` EMA with the `100` EMA slope rising by at least `0.02 ATR`, or a decisive bullish multi-year secular trend across the configured `260`, `520`, `1040`, and full-history windows
- no nearby long-term rolling resistance level within `0.75 ATR`, checked across medium, long, and full-history windows
- no structural resistance between entry and the final Fib target: horizontal pivot/rolling levels and validated long-term resistance trendlines both block the setup
- an ascending long-term support trendline below price reinforces the setup when it is close enough to matter
- or, a recent close broke above a configured horizontal rolling resistance level and remained within `0.75 ATR` of the broken level
- a descending corrective trendline built from structural pivot highs confirmed by the `5`-bar pivot window; the first anchor can come from any configured lookback up to full history, but the second anchor must be the latest pivot high in that window and no more than `35` bars old
- breakout confirmation: a recent high pierced the corrective trendline, a recent candle closed above the most recent corrective pivot high, and price closed above the prior recent peak from the selected structural window
- breakout validity: the bot evaluates the latest `3` candles and can alert a still-valid follow-through candle after the master candle
- breakout candle conviction: the recent confirmation window must include a bullish master candle; textbook patterns reinforce the breakout when present, but are not mandatory
- entry proximity: latest close must remain within `0.75 ATR` of the broken trendline so the alert does not chase a stale breakout
- target sanity: if the measured projection exceeds `6.0 ATR`, the setup is rejected unless fallback targets are explicitly enabled
- chart context: the primary corrective line, all validated nearby long-term lines relevant to entry or the Fib target path, and up to four nearby rolling/long-term support and resistance levels per side are drawn

Trade construction:

- entry: latest close
- stop: Trend-Based Fib Extension `0.0` level at corrective pivot $C$, using confirmed impulse low $A$, impulse high $B$, and corrective low $C$
- target 1: Trend-Based Fib Extension `1.0` level from $A \rightarrow B \rightarrow C$
- final target: Trend-Based Fib Extension `1.618` level from $A \rightarrow B \rightarrow C$
- management: checkpoint is Target 1; reward/risk uses the Excel formula, average of Target 1 % and Target 2 % divided by stop %
- minimum reward-to-risk: `1.5` for current multiday break-out scans

### Short setup

The logic is mirrored:

- bearish trend: fast EMA below slow EMA, or an emerging bearish breakout where price has lost the fast EMA with negative fast-EMA slope
- long-term trend: either close and slow EMA below the `100` EMA with the `100` EMA slope falling by at least `0.02 ATR`, or a decisive bearish multi-year secular trend across the configured `260`, `520`, `1040`, and full-history windows
- no nearby long-term rolling support level within `0.75 ATR`, checked across medium, long, and full-history windows
- no structural support between entry and the final Fib target: horizontal pivot/rolling levels and validated long-term support trendlines both block the setup, including descending support lines
- a descending long-term resistance trendline above price reinforces the setup when it is close enough to matter
- or, a recent close broke below a configured horizontal rolling support level and remained within `0.75 ATR` of the broken level
- an ascending corrective trendline built from structural pivot lows confirmed by the `5`-bar pivot window; the first anchor can come from any configured lookback up to full history, but the second anchor must be the latest pivot low in that window and no more than `35` bars old
- breakout confirmation: a recent low pierced the corrective trendline, a recent candle closed below the most recent corrective pivot low, and price closed below the prior recent trough from the selected structural window
- breakout validity: the bot evaluates the latest `3` candles and can alert a still-valid follow-through candle after the master candle
- breakout candle conviction: the recent confirmation window must include a bearish master candle; textbook patterns reinforce the breakout when present, but are not mandatory
- entry proximity: latest close must remain within `0.75 ATR` of the broken trendline so the alert does not chase a stale breakout
- target sanity: if the measured projection exceeds `6.0 ATR`, the setup is rejected unless fallback targets are explicitly enabled

Trade construction:

- entry: latest close
- stop: Trend-Based Fib Extension `0.0` level at corrective pivot $C$, using confirmed impulse high $A$, impulse low $B$, and corrective high $C$
- target 1: Trend-Based Fib Extension `1.0` level from $A \rightarrow B \rightarrow C$
- final target: Trend-Based Fib Extension `1.618` level from $A \rightarrow B \rightarrow C$
- management: checkpoint is Target 1; reward/risk uses the Excel formula, average of Target 1 % and Target 2 % divided by stop %
- minimum reward-to-risk: `1.5` for current multiday break-out scans

## Bollinger Reversion

Bollinger reversion is the intraday mean-reversion pattern.

Configured timeframe:

- `1h`

The pattern currently uses the named symbol list `intraday_focus` from `config/live-data.yaml`.

Current config values:

- `bb_period: 100`
- `bb_stddev: 2.0`
- `long_only: true`
- `master_candle_min_body_to_range: 0.15`
- `master_candle_strong_body_to_range: 0.55`
- `master_candle_close_zone: 0.25`
- `master_candle_uncertainty_body_max: 0.12`
- `master_candle_spinning_top_body_max: 0.25`
- `master_candle_uncertainty_wick_min: 0.30`
- `atr_filter_period: 5`
- `atr_filter_reference_lookback: 5`
- `atr_filter_max_multiplier: 1.5`
- `minimum_filter_checks: 1`
- `donchian_period: 100`
- `donchian_reference_lookback: 100`
- `stop_lookback: 5`
- `momentum_lookback: 10`
- `super_candle_lookback: 100`
- `super_candle_body_to_range_min: 0.2`
- `minimum_rr: 1.5`

Current live `intraday_focus` list:

- `SPY`
- `QQQ`
- `GLD`
- `ES`
- `NQ`
- `COR`
- `DD`
- `JPM`
- `SNDK`

### Long setup

The bot looks for:

- current close above the lower Bollinger Band
- bullish master candle after the lower-band reclaim; textbook candle patterns reinforce the setup when present but are not mandatory
- current or previous confirmed candle's low pierced the lower band
- current or previous momentum reading is positive
- at least `1` of the current or previous bar's textbook filters passes: ATR not above `1.5x` its value 5 bars earlier, the 100-bar Donchian channel not in a rising-trend state, or the bearish 100-bar super-candle filter

Trade construction:

- entry: latest close
- stop: lowest low over the last 5 bars
- target 1 and final target: Bollinger basis minus `0.3` standard deviations
- management: checkpoint is Target 1; reward/risk uses the same Excel calculator formula as other patterns
- trade is accepted only when reward-to-risk is at least `1.5`, because lower values receive no allocation from the Excel sizing model

### Short setup

The logic is mirrored:

- current close below the upper Bollinger Band
- bearish candle: close below open
- either current or previous candle pierced above the upper band
- current or previous momentum reading is negative

Trade construction:

- entry: latest close
- stop: highest high over the last 5 bars
- target 1 and final target: Bollinger basis plus `0.3` standard deviations
- management: checkpoint is Target 1; reward/risk uses the same Excel calculator formula as other patterns
- trade is accepted if reward-to-risk is positive

The current live configuration disables short entries for this pattern and keeps the intraday basket focused on liquid index, macro, and large-cap equity instruments. `SPY` remains the closest match to the course's stated long-only S&P 500 operating mode, while `QQQ`, `GLD`, `ES`, `NQ`, `COR`, `DD`, `JPM`, and `SNDK` broaden the opportunity set.

## Risk and Signal Output

For every accepted setup, the bot builds a `TradeSignal` that includes:

- pattern name
- direction
- instrument and timeframe
- candle timestamp
- entry, stop, and target
- checkpoint target and stop-adjustment guidance
- reward-to-risk ratio
- suggested position size based on live IBKR account value, the Excel R/R risk table, and the configured position-value cap
- explanatory notes

Excel risk allocation table from `Calcolatore Unico`:

- R/R below `1.5`: `0%`
- R/R from `1.5` through `5.0`: `2%`
- R/R above `5.0`: `3%`

The bot still applies `app.position_value_pct` as a maximum position-value cap after choosing the Excel risk allocation.

Signals are deduplicated so the same setup is not repeatedly sent for the same candle.

## Important Limitations

This is not a literal transcription of discretionary chart analysis. The following elements are deliberately approximated:

- support and resistance by rolling highs and lows
- reversal candles by simple wick/body and engulfing rules
- corrective trendline and pivot-boundary break built from recent confirmed pivot highs or lows
- Bollinger momentum confirmation by close difference over a fixed lookback

In other words, the system is best understood as a disciplined quantitative interpretation of the chart ideas, not a full replication of manual TradingView analysis.

## Current Live Operating Mode

The current production-like live setup reflects validation constraints found during IBKR testing:

- the broad watchlist remains enabled for higher-timeframe scans
- the intraday Bollinger scan is intentionally constrained via `symbol_lists.intraday_focus`
- this keeps the 1-hour IBKR path stable while preserving wider daily, weekly, and monthly coverage
