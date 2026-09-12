"""Scheduled price collector.

    airfare-collect run    --routes SJU-JFK,MIA-SJU --horizons 14,30,60
    airfare-collect run    --watchlist data/sample/watchlist.txt --export
    airfare-collect import --from data/observations

Every route x horizon is one provider request. ``--max-requests`` is a hard
budget so a misconfigured schedule can never drain an API quota.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from airfare.collect.partitions import export_day, import_partitions
from airfare.collect.watchlist import load_watchlist, parse_routes, plan_queries
from airfare.config import PROJECT_ROOT, get_settings
from airfare.providers import ProviderError, ProviderName, build_provider
from airfare.services.fx import FxService
from airfare.services.search import SearchService
from airfare.storage.sqlite import SqlitePriceHistory

log = logging.getLogger("airfare.collect")

DEFAULT_PARTITIONS = PROJECT_ROOT / "data" / "observations"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="airfare-collect", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="snapshot the watch-list into the history store")
    src = run.add_mutually_exclusive_group(required=True)
    src.add_argument("--routes", help="comma-separated, e.g. SJU-JFK,MIA-SJU")
    src.add_argument("--watchlist", type=Path, help="file with one route per line")
    run.add_argument(
        "--horizons", default="14,30,60", help="days ahead to search (comma-separated)"
    )
    run.add_argument(
        "--trip-length", type=int, default=7, help="round-trip length in days; 0 = one-way"
    )
    run.add_argument("--max-stops", type=int, default=2)
    run.add_argument(
        "--provider", default=ProviderName.SERPAPI.value, choices=[p.value for p in ProviderName]
    )
    run.add_argument("--max-requests", type=int, default=20, help="hard budget per run")
    run.add_argument("--top-n", type=int, default=None, help="observations to keep per query")
    run.add_argument("--export", action="store_true", help="also write today's CSV partition")
    run.add_argument("--partitions", type=Path, default=DEFAULT_PARTITIONS)
    run.add_argument("--dry-run", action="store_true", help="print the plan and exit")

    imp = sub.add_parser("import", help="load CSV partitions into the history store")
    imp.add_argument("--from", dest="src", type=Path, default=DEFAULT_PARTITIONS)
    return p


def _fmt_cheapest(usd: float | None) -> str:
    return f"${usd:,.0f}" if usd else "n/a"


def cmd_run(args: argparse.Namespace) -> int:
    settings = get_settings()
    routes = load_watchlist(args.watchlist) if args.watchlist else parse_routes(args.routes)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    today = datetime.now(UTC).date()
    queries = plan_queries(routes, horizons, today, args.trip_length or None, args.max_stops)

    if len(queries) > args.max_requests:
        log.error(
            "plan needs %d requests but --max-requests is %d; trim the plan or raise the budget",
            len(queries), args.max_requests,
        )  # fmt: skip
        return 2

    log.info(
        "plan: %d routes x %d horizons = %d requests via %s",
        len(routes),
        len(horizons),
        len(queries),
        args.provider,
    )
    for q in queries:
        log.info("  %s-%s dep %s ret %s", q.origin, q.destination, q.departure_date, q.return_date)
    if args.dry_run:
        return 0

    history = SqlitePriceHistory(settings.airfare_db_path)
    fx = FxService.from_settings(settings.airfare_db_path, settings.twelvedata_api_key)
    try:
        provider = build_provider(args.provider, settings, history)
    except ProviderError as e:
        log.error("%s", e)
        return 2
    service = SearchService(
        provider, fx, history, history_top_n=args.top_n or settings.history_top_n
    )

    saved = failed = 0
    for q in queries:
        try:
            result = service.search(q)
        except ProviderError as e:
            failed += 1
            log.warning("%s-%s %s failed: %s", q.origin, q.destination, q.departure_date, e)
            continue
        saved += result.observations_persisted
        log.info(
            "%s-%s %s: %d offers, cheapest %s, %d saved",
            q.origin, q.destination, q.departure_date, len(result.offers),
            _fmt_cheapest(result.cheapest.price.usd if result.cheapest else None),
            result.observations_persisted,
        )  # fmt: skip

    if args.export:
        export_day(history, today, args.partitions)
    log.info("done: %d observations saved, %d queries failed", saved, failed)
    return 1 if failed and not saved else 0


def cmd_import(args: argparse.Namespace) -> int:
    history = SqlitePriceHistory(get_settings().airfare_db_path)
    n = import_partitions(history, args.src)
    log.info("imported %d new observations from %s", n, args.src)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    return cmd_run(args) if args.command == "run" else cmd_import(args)


if __name__ == "__main__":
    sys.exit(main())
