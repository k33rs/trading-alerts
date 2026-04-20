from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from dataclasses import dataclass, field
import time
from typing import Protocol

import pandas as pd
import requests

from trading_bot.config import DataSourceConfig
from trading_bot.models import Instrument

BINANCE_INTERVALS = {
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d",
    "1wk": "1w",
    "1mo": "1M",
}

IBKR_INTERVALS = {
    "1h": "1 hour",
    "2h": "2 hours",
    "4h": "4 hours",
    "1d": "1 day",
    "1wk": "1 week",
    "1mo": "1 month",
}

IBKR_DURATIONS = {
    "1h": "365 D",
    "2h": "365 D",
    "4h": "365 D",
    "1d": "5 Y",
    "1wk": "10 Y",
    "1mo": "20 Y",
}


class HistoricalDataProvider(Protocol):
    def fetch_ohlcv(self, instrument: Instrument, interval: str) -> pd.DataFrame:
        ...

    def fetch_buying_power(self) -> float:
        ...

    def close(self) -> None:
        ...


@dataclass(slots=True)
class BinanceProvider:
    base_url: str = "https://api.binance.com"
    min_rows: int = 220
    timeout: int = 20

    def fetch_ohlcv(self, instrument: Instrument, interval: str) -> pd.DataFrame:
        symbol = (instrument.provider_symbol or instrument.symbol).replace("-", "").replace("/", "")
        binance_interval = BINANCE_INTERVALS.get(interval)
        if binance_interval is None:
            return pd.DataFrame()

        response = requests.get(
            f"{self.base_url.rstrip('/')}/api/v3/klines",
            params={"symbol": symbol, "interval": binance_interval, "limit": max(self.min_rows, 300)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload:
            return pd.DataFrame()

        frame = pd.DataFrame(
            payload,
            columns=[
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_asset_volume",
                "number_of_trades",
                "taker_buy_base_asset_volume",
                "taker_buy_quote_asset_volume",
                "ignore",
            ],
        )
        frame = frame[["open_time", "open", "high", "low", "close", "volume"]].copy()
        frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        for column in ["open", "high", "low", "close", "volume"]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna().set_index("open_time")
        frame = _drop_unclosed_latest_bar(frame, interval)
        if len(frame) < self.min_rows:
            return pd.DataFrame()
        return frame

    def close(self) -> None:
        return None

    def fetch_buying_power(self) -> float:
        raise RuntimeError("Buying power is only available from the IBKR provider")


class IbkrProvider:
    def __init__(
        self,
        host: str = "127.0.0.1",
        live_port: int = 4001,
        paper_port: int = 4002,
        account_mode: str = "live",
        profiles: dict[str, dict] | None = None,
        client_id: int = 7,
        read_only: bool = True,
        min_rows: int = 220,
        timeout: int = 20,
        default_what_to_show: str = "TRADES",
        default_use_rth: bool = False,
        request_pause_seconds: float = 0.5,
        auto_roll_futures: bool = False,
        min_days_to_expiry: int = 7,
    ) -> None:
        resolved_mode, resolved_profile = self._resolve_connection_profile(
            account_mode=account_mode,
            profiles=profiles,
            host=host,
            live_port=live_port,
            paper_port=paper_port,
            client_id=client_id,
            read_only=read_only,
            min_rows=min_rows,
            timeout=timeout,
            default_what_to_show=default_what_to_show,
            default_use_rth=default_use_rth,
            request_pause_seconds=request_pause_seconds,
            auto_roll_futures=auto_roll_futures,
            min_days_to_expiry=min_days_to_expiry,
        )
        self.host = str(resolved_profile["host"])
        self.live_port = int(resolved_profile["live_port"])
        self.paper_port = int(resolved_profile["paper_port"])
        self.account_mode = resolved_mode
        self.client_id = int(resolved_profile["client_id"])
        self.read_only = bool(resolved_profile["read_only"])
        self.min_rows = int(resolved_profile["min_rows"])
        self.timeout = int(resolved_profile["timeout"])
        self.default_what_to_show = str(resolved_profile["default_what_to_show"])
        self.default_use_rth = bool(resolved_profile["default_use_rth"])
        self.request_pause_seconds = float(resolved_profile["request_pause_seconds"])
        self.auto_roll_futures = bool(resolved_profile["auto_roll_futures"])
        self.min_days_to_expiry = int(resolved_profile["min_days_to_expiry"])
        self._ib = None
        self._util = None
        self._connect_error: RuntimeError | None = None
        self._last_request_at = 0.0
        self._historical_request_errors: list[tuple[int, str]] = []
        self._frame_cache: dict[tuple[str, str], pd.DataFrame] = {}

    @staticmethod
    def _resolve_connection_profile(
        *,
        account_mode: str,
        profiles: dict[str, dict] | None,
        host: str,
        live_port: int,
        paper_port: int,
        client_id: int,
        read_only: bool,
        min_rows: int,
        timeout: int,
        default_what_to_show: str,
        default_use_rth: bool,
        request_pause_seconds: float,
        auto_roll_futures: bool,
        min_days_to_expiry: int,
    ) -> tuple[str, dict]:
        base_profile = {
            "host": host,
            "live_port": live_port,
            "paper_port": paper_port,
            "client_id": client_id,
            "read_only": read_only,
            "min_rows": min_rows,
            "timeout": timeout,
            "default_what_to_show": default_what_to_show,
            "default_use_rth": default_use_rth,
            "request_pause_seconds": request_pause_seconds,
            "auto_roll_futures": auto_roll_futures,
            "min_days_to_expiry": min_days_to_expiry,
        }
        if not profiles:
            return account_mode, base_profile

        requested_mode = account_mode
        alias_map = {
            "live": "live_gateway",
            "paper": "paper_gateway",
        }
        profile_key = alias_map.get(requested_mode, requested_mode)
        profile = profiles.get(profile_key)
        if profile is None:
            available = ", ".join(sorted(profiles))
            raise ValueError(f"Unknown IBKR account_mode '{requested_mode}'. Available profiles: {available}")

        merged_profile = dict(base_profile)
        merged_profile.update(profile)
        resolved_mode = "paper" if profile_key.startswith("paper") else "live"
        return resolved_mode, merged_profile

    def fetch_ohlcv(self, instrument: Instrument, interval: str) -> pd.DataFrame:
        bar_size = IBKR_INTERVALS.get(interval)
        duration = IBKR_DURATIONS.get(interval)
        if bar_size is None or duration is None:
            return pd.DataFrame()

        request_key = (instrument.symbol, interval)
        cached_frame = self._frame_cache.get(request_key)
        if cached_frame is not None and _cached_frame_is_current(cached_frame, interval):
            return cached_frame.copy()

        bars = None
        for attempt in range(2):
            ib, util = self._connect()
            contract = self._build_contract(instrument)
            self._historical_request_errors.clear()
            try:
                self._throttle_requests()
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=instrument.what_to_show or self._default_what_to_show(instrument),
                    useRTH=instrument.use_rth if instrument.use_rth is not None else self.default_use_rth,
                    formatDate=2,
                    timeout=self.timeout,
                )
            except Exception as exc:
                if self._is_timeout_historical_data_error(exc):
                    raise RuntimeError(self._format_historical_data_error(contract, "Timed out requesting historical data")) from exc
                if not self._is_retryable_historical_data_error(exc):
                    raise RuntimeError(self._format_historical_data_error(contract, exc)) from exc
                self._drop_connection()
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise
            request_error = self._get_historical_request_error(contract)
            if request_error is not None:
                if self._is_timeout_historical_data_error(request_error):
                    raise RuntimeError(self._format_historical_data_error(contract, request_error))
                if not self._is_retryable_historical_data_error(request_error):
                    raise RuntimeError(self._format_historical_data_error(contract, request_error))
                self._drop_connection()
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise RuntimeError(self._format_historical_data_error(contract, request_error))
            if bars:
                break
            raise RuntimeError(self._format_historical_data_error(contract, "Timed out requesting historical data"))

        if not bars:
            return pd.DataFrame()
        frame = util.df(bars)
        if frame is None or frame.empty:
            return frame
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], utc=True)
            frame = frame.set_index("date")
        frame = frame.rename(columns=str.lower)
        frame = frame[[column for column in ["open", "high", "low", "close", "volume"] if column in frame.columns]]
        frame = frame.dropna().copy()
        frame = _drop_unclosed_latest_bar(frame, interval)
        if len(frame) < self.min_rows:
            return pd.DataFrame()
        self._frame_cache[request_key] = frame
        return frame.copy()

    def fetch_buying_power(self) -> float:
        ib, _ = self._connect()
        value = self._account_metric_value(ib.accountValues(), "BuyingPower")
        if value is None:
            value = self._account_metric_value(ib.accountSummary(), "BuyingPower")
        if value is not None:
            return value
        raise RuntimeError("IBKR account updates did not include a positive BuyingPower value")

    @staticmethod
    def _account_metric_value(account_values, tag: str) -> float | None:
        candidates = [item for item in account_values if item.tag == tag]
        candidates.sort(key=lambda item: item.currency != "BASE")
        for item in candidates:
            try:
                value = float(item.value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    def scan_instruments(self, source: dict) -> list[Instrument]:
        self._ensure_event_loop()
        try:
            from ib_insync import ScannerSubscription
        except ImportError as exc:
            raise RuntimeError("ib_insync is required for IBKR scanner discovery.") from exc

        subscription_kwargs: dict[str, object] = {
            "instrument": str(source.get("instrument", "STK")),
            "locationCode": str(source.get("location_code", "STK.US.MAJOR")),
            "scanCode": str(source.get("scan_code", "MOST_ACTIVE")),
            "numberOfRows": int(source.get("rows", 25)),
        }
        if source.get("min_price") is not None:
            subscription_kwargs["abovePrice"] = float(source["min_price"])
        if source.get("min_volume") is not None:
            subscription_kwargs["aboveVolume"] = int(source["min_volume"])

        scan_results = None
        for attempt in range(2):
            try:
                ib, _ = self._connect()
                self._throttle_requests()
                scan_results = ib.reqScannerData(ScannerSubscription(**subscription_kwargs))
                break
            except Exception:
                self._drop_connection()
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise
        if scan_results is None:
            return []
        instruments: list[Instrument] = []
        source_name = str(source.get("name", source.get("scan_code", "ibkr_scanner")))
        source_priority = max(0, int(source.get("priority", 100)))
        allowed_stock_types = {str(value).upper() for value in source.get("allowed_stock_types", [])}
        for result in scan_results:
            contract = result.contractDetails.contract
            asset_type = _ibkr_asset_type(getattr(contract, "secType", ""))
            if asset_type is None:
                continue
            stock_type = str(getattr(result.contractDetails, "stockType", "")).upper()
            if asset_type == "stock" and allowed_stock_types and stock_type not in allowed_stock_types:
                continue
            symbol = str(getattr(contract, "symbol", "")).strip()
            if not symbol:
                continue
            instruments.append(
                Instrument(
                    symbol=symbol,
                    label=symbol,
                    asset_name=str(getattr(result.contractDetails, "longName", "") or symbol),
                    provider="ibkr",
                    provider_symbol=symbol,
                    asset_type=asset_type,
                    exchange=str(getattr(contract, "exchange", "") or "SMART"),
                    primary_exchange=str(getattr(contract, "primaryExchange", "") or "") or None,
                    currency=str(getattr(contract, "currency", "") or "USD"),
                    discovery_source=source_name,
                    discovery_priority=source_priority,
                    discovery_rank=max(0, int(getattr(result, "rank", len(instruments)))),
                )
            )
        return instruments

    def _connect(self):
        if self._ib is not None and self._ib.isConnected():
            return self._ib, self._util
        if self._connect_error is not None:
            raise self._connect_error

        self._ensure_event_loop()

        try:
            from ib_insync import IB, CFD, Contract, Forex, Future, Index, Stock, util
        except ImportError as exc:
            raise RuntimeError(
                "ib_insync is required for the IBKR provider. Install project dependencies or pip install ib-insync."
            ) from exc

        self._contract_types = {
            "stock": Stock,
            "future": Future,
            "forex": Forex,
            "index": Index,
            "cfd": CFD,
            "contract": Contract,
        }
        ib = IB()
        try:
            ib.connect(self.host, self._selected_port(), clientId=self.client_id, readonly=self.read_only)
        except Exception as exc:
            self._connect_error = RuntimeError(
                f"IBKR {self.account_mode} connection failed at {self.host}:{self._selected_port()}. Start TWS or IB Gateway and enable the API port."
            )
            raise self._connect_error from exc
        ib.errorEvent += self._on_ib_error
        self._ib = ib
        self._util = util
        self._connect_error = None
        return self._ib, self._util

    def _drop_connection(self) -> None:
        if self._ib is not None:
            try:
                self._ib.errorEvent -= self._on_ib_error
            except Exception:
                pass
            try:
                if self._ib.isConnected():
                    self._ib.disconnect()
            except Exception:
                pass
        self._ib = None
        self._util = None
        self._connect_error = None
        self._historical_request_errors.clear()

    def _throttle_requests(self) -> None:
        if self.request_pause_seconds <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = self.request_pause_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _selected_port(self) -> int:
        return self.paper_port if self.account_mode == "paper" else self.live_port

    def _ensure_event_loop(self) -> None:
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

    def _build_contract(self, instrument: Instrument):
        asset_type = (instrument.asset_type or "stock").lower()
        symbol = instrument.provider_symbol or instrument.symbol
        exchange = instrument.exchange or "SMART"
        currency = instrument.currency or "USD"

        if asset_type == "forex":
            base_quote = symbol.replace("/", "")
            return self._contract_types["forex"](base_quote)
        if asset_type == "future":
            if self.auto_roll_futures:
                resolved_contract = self._resolve_future_contract(symbol, exchange, currency, instrument)
                if resolved_contract is not None:
                    return resolved_contract
            return self._contract_types["future"](
                symbol=symbol,
                lastTradeDateOrContractMonth=instrument.last_trade_date_or_contract_month or "",
                exchange=exchange,
                currency=currency,
            )
        if asset_type == "index":
            return self._contract_types["index"](symbol=symbol, exchange=exchange, currency=currency)
        if asset_type == "cfd":
            return self._contract_types["cfd"](symbol=symbol, exchange=exchange, currency=currency)
        if asset_type == "stock":
            kwargs = {"symbol": symbol, "exchange": exchange, "currency": currency}
            if instrument.primary_exchange:
                kwargs["primaryExchange"] = instrument.primary_exchange
            return self._contract_types["stock"](**kwargs)

        contract = self._contract_types["contract"]()
        contract.symbol = symbol
        contract.secType = asset_type.upper()
        contract.exchange = exchange
        contract.currency = currency
        if instrument.primary_exchange:
            contract.primaryExchange = instrument.primary_exchange
        if instrument.last_trade_date_or_contract_month:
            contract.lastTradeDateOrContractMonth = instrument.last_trade_date_or_contract_month
        return contract

    def _default_what_to_show(self, instrument: Instrument) -> str:
        if (instrument.asset_type or "").lower() == "forex":
            return "MIDPOINT"
        return self.default_what_to_show

    @staticmethod
    def _is_retryable_historical_data_error(exc: Exception | str) -> bool:
        message = str(exc).lower()
        non_retryable_markers = (
            "error 162",
            "historical market data service error",
            "connected from a different ip address",
            "no market data permissions",
            "no security definition",
        )
        if any(marker in message for marker in non_retryable_markers):
            return False
        retryable_markers = (
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "not connected",
            "peer closed connection",
        )
        return any(marker in message for marker in retryable_markers)

    @staticmethod
    def _is_timeout_historical_data_error(exc: Exception | str) -> bool:
        message = str(exc).lower()
        timeout_markers = (
            "timeout",
            "timed out",
        )
        return any(marker in message for marker in timeout_markers)

    @staticmethod
    def _format_historical_data_error(contract, exc: Exception) -> str:
        message = " ".join(str(exc).split())
        contract_label = getattr(contract, "localSymbol", "") or getattr(contract, "symbol", "") or repr(contract)
        sec_type = getattr(contract, "secType", "")
        exchange = getattr(contract, "exchange", "")
        primary_exchange = getattr(contract, "primaryExchange", "")
        currency = getattr(contract, "currency", "")
        venue = "/".join(part for part in (str(sec_type), str(exchange), str(primary_exchange), str(currency)) if part)
        suffix = f" ({venue})" if venue else ""
        return f"Historical data request failed for {contract_label}{suffix}: {message}"

    def _on_ib_error(self, req_id: int, error_code: int, error_string: str, contract) -> None:
        if req_id <= 0:
            return
        self._historical_request_errors.append((error_code, error_string))

    def _get_historical_request_error(self, contract) -> str | None:
        if not self._historical_request_errors:
            return None
        non_info_errors = [
            (error_code, error_string)
            for error_code, error_string in self._historical_request_errors
            if error_code not in {2104, 2106, 2158}
        ]
        if not non_info_errors:
            return None
        error_code, error_string = non_info_errors[-1]
        return f"Error {error_code}: {error_string}"

    def _resolve_future_contract(
        self,
        symbol: str,
        exchange: str,
        currency: str,
        instrument: Instrument,
    ):
        ib, _ = self._connect()
        explicit_month = instrument.last_trade_date_or_contract_month
        if explicit_month:
            return self._contract_types["future"](
                symbol=symbol,
                lastTradeDateOrContractMonth=explicit_month,
                exchange=exchange,
                currency=currency,
            )

        base_contract = self._contract_types["future"](
            symbol=symbol,
            lastTradeDateOrContractMonth="",
            exchange=exchange,
            currency=currency,
        )
        self._throttle_requests()
        details = ib.reqContractDetails(base_contract)
        if not details:
            return None

        selected = self._select_active_future_contract(details, instrument.min_days_to_expiry or self.min_days_to_expiry)
        if selected is None:
            return None
        _apply_resolved_future_identity(instrument, selected)
        return selected.contract

    def _select_active_future_contract(self, details: list, min_days_to_expiry: int):
        threshold = date.today() + timedelta(days=max(0, min_days_to_expiry))
        candidates: list[tuple[date, object]] = []
        fallback_candidates: list[tuple[date, object]] = []
        for detail in details:
            expiry = self._parse_expiry_date(getattr(detail.contract, "lastTradeDateOrContractMonth", ""))
            if expiry is None:
                continue
            fallback_candidates.append((expiry, detail))
            if expiry >= threshold:
                candidates.append((expiry, detail))

        pool = candidates or fallback_candidates
        if not pool:
            return None
        pool.sort(key=lambda item: item[0])
        return pool[0][1]

    @staticmethod
    def _parse_expiry_date(raw_value: str) -> date | None:
        if not raw_value:
            return None
        digits_only = "".join(character for character in raw_value.strip() if character.isdigit())
        if len(digits_only) >= 8:
            try:
                return datetime.strptime(digits_only[:8], "%Y%m%d").date()
            except ValueError:
                pass
        if len(digits_only) >= 6:
            try:
                parsed = datetime.strptime(digits_only[:6], "%Y%m")
            except ValueError:
                return None
            return date(parsed.year, parsed.month, 1)
        return None

    def close(self) -> None:
        self._drop_connection()


def _apply_resolved_future_identity(instrument: Instrument, detail: object) -> None:
    contract = getattr(detail, "contract", None)
    if contract is None:
        return
    market_name = str(getattr(detail, "longName", "") or getattr(detail, "marketName", "") or instrument.asset_name).strip()
    local_symbol = str(getattr(contract, "localSymbol", "")).strip()
    exchange = str(getattr(contract, "exchange", "") or instrument.exchange or "").strip()
    expiry = _future_expiry_label(str(getattr(contract, "lastTradeDateOrContractMonth", "")))
    identity_parts = [market_name]
    if local_symbol:
        identity_parts.append(f"contract {local_symbol}")
    if exchange:
        identity_parts.append(exchange)
    if expiry:
        identity_parts.append(f"expires {expiry}")
    instrument.asset_name = " | ".join(identity_parts)


def _future_expiry_label(raw_value: str) -> str:
    digits_only = "".join(character for character in raw_value if character.isdigit())
    if len(digits_only) < 6:
        return ""
    try:
        return datetime.strptime(digits_only[:6], "%Y%m").strftime("%b %Y")
    except ValueError:
        return raw_value.strip()


@dataclass(slots=True)
class MarketDataRouter:
    config: DataSourceConfig
    account_mode: str | None = None
    client_id: int | None = None
    _providers: dict[str, HistoricalDataProvider] = field(init=False)

    def __post_init__(self) -> None:
        provider_settings = self.config.providers
        ibkr_settings = dict(provider_settings.get("ibkr", {}))
        if self.account_mode is not None:
            ibkr_settings["account_mode"] = self.account_mode
        if self.client_id is not None:
            ibkr_settings["client_id"] = self.client_id
        self._providers = {
            "binance": BinanceProvider(**provider_settings.get("binance", {})),
            "ibkr": IbkrProvider(**ibkr_settings),
        }

    def fetch_ohlcv(self, instrument: Instrument, interval: str) -> pd.DataFrame:
        provider_name = (instrument.provider or self.config.default_provider).lower()
        provider = self._providers.get(provider_name)
        if provider is None:
            raise ValueError(f"Unsupported provider: {provider_name}")
        return provider.fetch_ohlcv(instrument, interval)

    def fetch_buying_power(self) -> float:
        provider = self._providers.get("ibkr")
        if provider is None:
            raise RuntimeError("IBKR provider is required to read buying power")
        return provider.fetch_buying_power()

    def scan_instruments(self, source: dict) -> list[Instrument]:
        provider_name = str(source.get("provider", self.config.default_provider)).lower()
        provider = self._providers.get(provider_name)
        if not isinstance(provider, IbkrProvider):
            raise ValueError(f"Dynamic universe discovery is unsupported for provider: {provider_name}")
        return provider.scan_instruments(source)

    def close(self) -> None:
        for provider in self._providers.values():
            provider.close()


def _drop_unclosed_latest_bar(frame: pd.DataFrame, interval: str, now: datetime | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    latest = pd.Timestamp(frame.index[-1])
    if pd.isna(latest):
        return frame

    now_timestamp = pd.Timestamp(now or datetime.now(timezone.utc))
    if latest.tzinfo is None:
        latest_utc = latest.tz_localize(timezone.utc)
    else:
        latest_utc = latest.tz_convert(timezone.utc)
    if now_timestamp.tzinfo is None:
        now_utc = now_timestamp.tz_localize(timezone.utc)
    else:
        now_utc = now_timestamp.tz_convert(timezone.utc)

    close_time = _bar_close_time(latest_utc, interval)
    if close_time is None or close_time <= now_utc:
        return frame
    return frame.iloc[:-1].copy()


def confirmed_ohlcv_frame(frame: pd.DataFrame, interval: str, now: datetime | None = None) -> pd.DataFrame:
    required_columns = [column for column in ["open", "high", "low", "close", "volume"] if column in frame.columns]
    if not required_columns:
        return pd.DataFrame()
    complete = frame.dropna(subset=required_columns).copy()
    return _drop_unclosed_latest_bar(complete, interval, now)


def _cached_frame_is_current(frame: pd.DataFrame, interval: str, now: datetime | None = None) -> bool:
    if frame.empty:
        return False
    latest = pd.Timestamp(frame.index[-1])
    if pd.isna(latest):
        return False
    if latest.tzinfo is None:
        latest = latest.tz_localize(timezone.utc)
    else:
        latest = latest.tz_convert(timezone.utc)
    current_time = pd.Timestamp(now or datetime.now(timezone.utc))
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize(timezone.utc)
    else:
        current_time = current_time.tz_convert(timezone.utc)
    latest_close = _bar_close_time(latest, interval)
    next_close = _bar_close_time(latest_close, interval) if latest_close is not None else None
    return next_close is not None and current_time < next_close


def _bar_close_time(open_time: pd.Timestamp, interval: str) -> pd.Timestamp | None:
    if interval.endswith("h"):
        try:
            hours = int(interval[:-1])
        except ValueError:
            return None
        if hours <= 0:
            return None
        return open_time + pd.Timedelta(hours=hours)
    if interval == "1d":
        return open_time + pd.Timedelta(days=1)
    if interval == "1wk":
        return open_time + pd.Timedelta(weeks=1)
    if interval == "1mo":
        return open_time + pd.DateOffset(months=1)
    return None


def _ibkr_asset_type(sec_type: str) -> str | None:
    asset_types = {
        "STK": "stock",
        "FUT": "future",
        "IND": "index",
        "CASH": "forex",
        "CFD": "cfd",
    }
    return asset_types.get(sec_type.upper())