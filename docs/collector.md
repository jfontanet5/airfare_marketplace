# Price collector

`airfare-collect` snapshots a watch-list of routes into the history store so the
price-drop model can eventually train on real observations instead of synthetic ones.

```
airfare-collect run --watchlist data/sample/watchlist.txt --export   # 3 routes × 3 horizons = 9 requests
airfare-collect run --routes SJU-JFK --horizons 21 --dry-run           # show the plan, spend nothing
airfare-collect import --from data/observations                        # rebuild SQLite from CSV partitions
```

Every route × horizon is one provider request, and `--max-requests` (default 20) is a hard
stop, so a misconfigured schedule cannot drain the SerpApi quota. Horizons default to
14, 30 and 60 days out with a 7-day round trip; `--trip-length 0` collects one-way fares.

## Why CSV partitions

The SQLite file is local and gitignored. `--export` also writes the day's observations to
`data/observations/<YYYY-MM-DD>.csv` — a few KB per day, diff-friendly, and committable.
`import` is idempotent (the store enforces `(signature, search_ts)` uniqueness), so a
fresh clone rebuilds its history with one command. This is what lets a scheduled job
persist results without any external database.

## Scheduling

**GitHub Actions** — `.github/workflows/collect.yml` runs on manual dispatch; uncomment
the `schedule` block for daily runs. Add `SERPAPI_API_KEY` under *Settings → Secrets and
variables → Actions*. Each run imports existing partitions, collects, and commits the new
partition.

**Local (macOS launchd)** — copy `docs/com.airfare.collect.plist` to
`~/Library/LaunchAgents/`, edit the two absolute paths, then:

```
launchctl load ~/Library/LaunchAgents/com.airfare.collect.plist
```

It runs `make collect` daily at 08:00 local time while the machine is awake; logs land in
`/tmp/airfare-collect.log`.

## Training on the result

Labels need a *future* observation for the same itinerary, so rows from the last 7 days
are always censored. With the sample watch-list (~9 queries × top-30 offers per day), the
`MIN_OBSERVATION_ROWS` threshold of 300 labeled rows is typically reached after 3–4 weeks:

```
make train-real        # airfare-train --source observations
```

The model card (`models/price_drop/card.json`) records `data_source: observations`, the
date range, row counts and validation metrics, and the UI drops the "synthetic demo" label.
