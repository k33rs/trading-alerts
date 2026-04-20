from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv
import yaml

from trading_bot.models import Instrument


SUPPORTED_PROVIDERS = {"ibkr", "binance"}
IBKR_EXCHANGE_REQUIRED_ASSET_TYPES = {"stock", "future", "index", "cfd", "contract"}


@dataclass(slots=True)
class AppConfig:
    scan_interval_seconds: int
    risk_per_trade_pct: float
    position_value_pct: float = 0.01
    historical_request_budget: int = 20
    historical_failure_retry_seconds: int = 3_600
    timeframe_refresh_seconds: dict[str, int] | None = None


@dataclass(slots=True)
class TelegramConfig:
    enabled: bool
    bot_token: str | None
    chat_id: str | None


@dataclass(slots=True)
class PatternConfig:
    name: str
    enabled: bool
    settings: dict


@dataclass(slots=True)
class DataSourceConfig:
    default_provider: str
    providers: dict[str, dict]


@dataclass(slots=True)
class UniverseConfig:
    enabled: bool
    database_path: str
    client_id: int | None
    snapshot_max_age_seconds: int
    core_symbols: list[str]
    watchlist_candidates: list[str]
    watchlist_candidate_limit: int
    max_dynamic_candidates: int
    sources: list[dict]
    trend_filter: dict


@dataclass(slots=True)
class BotConfig:
    app: AppConfig
    telegram: TelegramConfig
    data_source: DataSourceConfig
    universe: UniverseConfig
    watchlist: list[Instrument]
    symbol_lists: dict[str, list[str]]
    patterns: dict[str, PatternConfig]


def load_config(config_path: Path) -> BotConfig:
    load_dotenv()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    app = AppConfig(**raw["app"])
    telegram_raw = raw["telegram"]
    telegram = TelegramConfig(
        enabled=telegram_raw.get("enabled", False),
        bot_token=os.getenv(telegram_raw.get("bot_token_env", "TELEGRAM_BOT_TOKEN")),
        chat_id=os.getenv(telegram_raw.get("chat_id_env", "TELEGRAM_CHAT_ID")),
    )
    data_source_raw = raw.get("data_source", {})
    providers = data_source_raw.get("providers", {})
    ibkr_host_override = os.getenv("IBKR_HOST")
    if ibkr_host_override and isinstance(providers.get("ibkr"), dict):
        providers = dict(providers)
        ibkr_provider = dict(providers["ibkr"])
        profiles = {
            name: {**profile, "host": ibkr_host_override}
            for name, profile in ibkr_provider.get("profiles", {}).items()
        }
        ibkr_provider["host"] = ibkr_host_override
        ibkr_provider["profiles"] = profiles
        providers["ibkr"] = ibkr_provider
    data_source = DataSourceConfig(
        default_provider=data_source_raw.get("default_provider", "ibkr"),
        providers=providers,
    )
    watchlist = [Instrument(**item) for item in raw["watchlist"]]
    symbol_lists = {name: list(symbols) for name, symbols in raw.get("symbol_lists", {}).items()}
    _validate_watchlist(watchlist, data_source, symbol_lists)
    universe_raw = raw.get("universe", {})
    universe = UniverseConfig(
        enabled=bool(universe_raw.get("enabled", False)),
        database_path=str(universe_raw.get("database_path", "data/universe.db")),
        client_id=int(universe_raw["client_id"]) if universe_raw.get("client_id") is not None else None,
        snapshot_max_age_seconds=max(0, int(universe_raw.get("snapshot_max_age_seconds", 21_600))),
        core_symbols=list(universe_raw.get("core_symbols", [])),
        watchlist_candidates=list(universe_raw.get("watchlist_candidates", [])),
        watchlist_candidate_limit=max(0, int(universe_raw.get("watchlist_candidate_limit", 0))),
        max_dynamic_candidates=int(universe_raw.get("max_dynamic_candidates", 0)),
        sources=list(universe_raw.get("sources", [])),
        trend_filter=dict(universe_raw.get("trend_filter", {})),
    )
    unknown_core_symbols = sorted(set(universe.core_symbols) - {instrument.symbol for instrument in watchlist})
    if unknown_core_symbols:
        joined_symbols = ", ".join(unknown_core_symbols)
        raise ValueError(f"Universe core_symbols contains unknown watchlist symbols: {joined_symbols}.")
    unknown_watchlist_candidates = sorted(set(universe.watchlist_candidates) - {instrument.symbol for instrument in watchlist})
    if unknown_watchlist_candidates:
        joined_symbols = ", ".join(unknown_watchlist_candidates)
        raise ValueError(f"Universe watchlist_candidates contains unknown watchlist symbols: {joined_symbols}.")
    patterns = {
        name: PatternConfig(name=name, enabled=settings.get("enabled", False), settings=settings)
        for name, settings in raw["patterns"].items()
    }
    for name, pattern in patterns.items():
        symbol_list_name = pattern.settings.get("symbol_list")
        if symbol_list_name and symbol_list_name not in symbol_lists:
            raise ValueError(f"Pattern '{name}' references unknown symbol_list '{symbol_list_name}'.")
    return BotConfig(
        app=app,
        telegram=telegram,
        data_source=data_source,
        universe=universe,
        watchlist=watchlist,
        symbol_lists=symbol_lists,
        patterns=patterns,
    )


def _validate_watchlist(
    watchlist: list[Instrument],
    data_source: DataSourceConfig,
    symbol_lists: dict[str, list[str]],
) -> None:
    if not watchlist:
        raise ValueError("Watchlist cannot be empty.")

    symbols: set[str] = set()
    for instrument in watchlist:
        symbol = instrument.symbol.strip()
        if not symbol:
            raise ValueError("Watchlist instruments must define a non-empty symbol.")
        if symbol in symbols:
            raise ValueError(f"Duplicate watchlist symbol '{symbol}'.")
        symbols.add(symbol)

        provider_name = (instrument.provider or data_source.default_provider).lower()
        if provider_name not in SUPPORTED_PROVIDERS:
            raise ValueError(f"Instrument '{symbol}' uses unsupported provider '{provider_name}'.")

        asset_type = (instrument.asset_type or "stock").lower()
        if (
            provider_name == "ibkr"
            and asset_type in IBKR_EXCHANGE_REQUIRED_ASSET_TYPES
            and not instrument.exchange
        ):
            raise ValueError(f"Instrument '{symbol}' requires an exchange for IBKR {asset_type} data.")

    for list_name, list_symbols in symbol_lists.items():
        unknown_symbols = sorted(set(list_symbols) - symbols)
        if unknown_symbols:
            joined_symbols = ", ".join(unknown_symbols)
            raise ValueError(f"Symbol list '{list_name}' contains unknown watchlist symbols: {joined_symbols}.")