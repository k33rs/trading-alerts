from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import time

from trading_bot.config import load_config
from trading_bot.data_provider import MarketDataRouter
from trading_bot.main import configure_logging, log
from trading_bot.universe import UniverseStore, refresh_universe, universe_instrument_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic trading universe refresh service")
    parser.add_argument("--config", default="config/live-data.yaml", help="Path to YAML config")
    parser.add_argument(
        "--account-mode",
        choices=["live", "paper", "live_gateway", "paper_gateway", "live_tws", "paper_tws"],
        help="Select the IBKR connection profile to use",
    )
    parser.add_argument("--once", action="store_true", help="Refresh once and exit")
    parser.add_argument("--list", action="store_true", help="Print the current persisted hot-instrument snapshot as JSON")
    parser.add_argument("--source", help="Filter --list results by discovery source")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of --list results; 0 prints all")
    parser.add_argument("--interval-seconds", type=int, default=14_400, help="Refresh cadence for service mode")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_config(Path(args.config))
    if not config.universe.enabled:
        raise RuntimeError("Universe service is disabled in configuration.")
    if args.list:
        instruments = UniverseStore(Path(config.universe.database_path)).load()
        if args.source:
            instruments = [instrument for instrument in instruments if instrument.discovery_source == args.source]
        if args.limit > 0:
            instruments = instruments[: args.limit]
        print(json.dumps([universe_instrument_payload(instrument) for instrument in instruments], indent=2))
        return
    provider = MarketDataRouter(
        config.data_source,
        account_mode=args.account_mode,
        client_id=config.universe.client_id,
    )
    try:
        while True:
            result = refresh_universe(config, provider)
            source_summary = ", ".join(f"{name}={count}" for name, count in result.source_counts.items())
            outcome = "retained previous snapshot" if result.retained_previous_snapshot else "published"
            log(f"Universe refresh: {len(result.instruments)} instruments ({source_summary}; {outcome}).")
            for failure in result.failures:
                log(f"Universe source failure: {failure}")
            if args.once:
                return
            next_refresh = datetime.now() + timedelta(seconds=args.interval_seconds)
            log(f"Sleeping for {args.interval_seconds} seconds. Next universe refresh at {next_refresh.strftime('%Y-%m-%d %H:%M:%S')}.")
            time.sleep(args.interval_seconds)
    finally:
        provider.close()


if __name__ == "__main__":
    main()