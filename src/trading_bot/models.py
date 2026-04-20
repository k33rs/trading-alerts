from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Instrument:
    symbol: str
    label: str
    asset_name: str
    provider: str | None = None
    provider_symbol: str | None = None
    asset_type: str | None = None
    exchange: str | None = None
    currency: str | None = None
    primary_exchange: str | None = None
    last_trade_date_or_contract_month: str | None = None
    min_days_to_expiry: int | None = None
    what_to_show: str | None = None
    use_rth: bool | None = None
    discovery_source: str | None = None
    discovery_priority: int = 100
    discovery_rank: int = 999
    trend_direction: str | None = None
    trend_strength: float | None = None
    daily_trend_context: str | None = None


@dataclass(slots=True)
class TradeSignal:
    pattern: str
    direction: str
    timeframe: str
    instrument: Instrument
    candle_time: str
    entry: float
    stop_loss: float
    target: float
    rr: float
    position_size: float | None
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dedupe_key(self) -> str:
        return "|".join(
            [
                self.instrument.symbol,
                self.timeframe,
                self.pattern,
                self.direction,
                self.candle_time,
            ]
        )

    def to_telegram_message(self) -> str:
        note_lines = "\n".join(f"- {note}" for note in self.notes)
        size_line = (
            f"Suggested size: {self.position_size:.4f}\n" if self.position_size is not None else ""
        )
        management_line = _format_management_plan(self.metadata)
        return (
            f"{self.pattern} {self.direction.upper()} on {self.instrument.label} ({self.instrument.asset_name})\n"
            f"Timeframe: {self.timeframe}\n"
            f"Candle: {self.candle_time}\n"
            f"Entry: {self.entry:.4f}\n"
            f"Stop: {self.stop_loss:.4f}\n"
            f"Target: {self.target:.4f}\n"
            f"RR: {self.rr:.2f}\n"
            f"{size_line}"
            f"{management_line}"
            f"Notes:\n{note_lines}"
        ).strip()


def _format_management_plan(metadata: dict[str, Any]) -> str:
    management = metadata.get("management")
    if not isinstance(management, dict):
        return ""

    lines: list[str] = []
    checkpoint = management.get("checkpoint_target")
    final_target = management.get("final_target")
    risk_allocation_pct = management.get("risk_allocation_pct")
    risk_model = management.get("risk_model")
    stop_adjustment = management.get("stop_adjustment")
    if isinstance(checkpoint, (int, float)):
        lines.append(f"Checkpoint: {checkpoint:.4f}")
    if isinstance(final_target, (int, float)):
        lines.append(f"Final target: {final_target:.4f}")
    if isinstance(risk_allocation_pct, (int, float)):
        suffix = f" ({risk_model})" if isinstance(risk_model, str) and risk_model else ""
        lines.append(f"Risk allocation: {risk_allocation_pct * 100:.1f}%{suffix}")
    if isinstance(stop_adjustment, str) and stop_adjustment:
        lines.append(f"Stop management: {stop_adjustment}")
    if not lines:
        return ""
    return "\n".join(lines) + "\n"