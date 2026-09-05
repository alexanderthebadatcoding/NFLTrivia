#!/usr/bin/env python3
"""
espn_roster_stats_fast.py

Same behavior/output as espn_roster_stats.py, but with two changes aimed at
making it run fast on a normal laptop:

  1. A shared requests.Session with a pooled HTTPAdapter, so we reuse TCP/TLS
     connections instead of opening a new one for every single GET (urllib's
     urlopen() does not pool connections at all).

  2. Per-team athlete processing (resolve athlete -> fetch stats) runs on a
     ThreadPoolExecutor instead of a plain for-loop. This is a purely I/O-bound
     workload (waiting on ESPN's servers), so threads are the right tool here,
     not multiprocessing -- Python releases the GIL during network I/O, so you
     get real concurrency without the overhead of separate processes.

Tune concurrency with --workers (default 16). Roster fetching per team/year is
still sequential (it's 1-2 calls per team/year, not worth parallelizing), but
the athlete resolve+stats fetch -- the part that scales with roster size -- is
now concurrent.

Usage is identical to the original script, plus:
    --workers N     number of concurrent athlete-fetch threads (default: 16)

Example:
    python espn_roster_stats_fast.py 2025 --divisions "NFC North" --workers 20
"""

import argparse
import collections
import math
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import nfl_data_py as nfl_data_py
except Exception:  # pragma: no cover - optional dependency
    nfl_data_py = None

import requests
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry
except ImportError:
    from requests.packages.urllib3.util.retry import Retry

BASE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

DEFAULT_ALL_START = 2025
DEFAULT_ALL_END = 2016
DEFAULT_WORKERS = 16
DEFAULT_MAX_RPS = 8  # global cap on requests/second across ALL threads combined


class RateLimiter:
    """
    Caps total requests/second across every worker thread, independent of how
    many threads are running. --workers controls how many requests can be
    *in flight* at once; this controls how many can *start* per second --
    the thing ESPN's edge/WAF actually seems to react to (bursty concurrent
    connections from one IP), not sustained parallelism.
    """
    def __init__(self, max_per_second):
        self.max_per_second = max_per_second
        self.lock = threading.Lock()
        self.timestamps = collections.deque()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            while self.timestamps and now - self.timestamps[0] > 1.0:
                self.timestamps.popleft()
            if len(self.timestamps) >= self.max_per_second:
                sleep_for = 1.0 - (now - self.timestamps[0])
                if sleep_for > 0:
                    time.sleep(sleep_for)
            self.timestamps.append(time.monotonic())


_rate_limiter = RateLimiter(DEFAULT_MAX_RPS)

DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"],
    "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"],
    "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WSH"],
    "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"],
    "NFC West": ["ARI", "LAR", "SF", "SEA"],
}


def normalize_division_name(s):
    return re.sub(r"[\s_-]+", " ", s.strip()).upper()


DIVISION_LOOKUP = {normalize_division_name(k): k for k in DIVISIONS}
TEAM_TO_DIVISION = {team: div for div, teams in DIVISIONS.items() for team in teams}


def teams_for_division(name):
    key = normalize_division_name(name)
    if key not in DIVISION_LOOKUP:
        valid = ", ".join(DIVISIONS.keys())
        raise ValueError(f"Unknown division '{name}'. Valid divisions: {valid}")
    return DIVISIONS[DIVISION_LOOKUP[key]]


POSITION_GROUP_MAP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}

WANTED_KEYS_BY_GROUP = {
    "QB": {"yds", "td", "int"},
    "RB": {"ryds", "rtd", "recyds", "rectd", "rec"},
    "WR": {"recyds", "rectd", "rec"},
    "TE": {"recyds", "rectd", "rec"},
}

STAT_KEY_MAP = {
    "passingYards": "yds",
    "passingTouchdowns": "td",
    "interceptions": "int",
    "rushingYards": "ryds",
    "rushingTouchdowns": "rtd",
    "receivingYards": "recyds",
    "receivingTouchdowns": "rectd",
    "receptions": "rec",
}

TIMEOUT = 15
ATHLETE_ID_PATTERN = re.compile(r"/athletes/(\d+)")

# ---------------------------------------------------------------------------
# Pooled session: this is the first big speedup. One Session + HTTPAdapter
# reuses TCP connections (and TLS sessions) across requests instead of
# renegotiating a fresh connection per call. pool_maxsize should be >= your
# thread count or threads will queue waiting for a free connection.
# ---------------------------------------------------------------------------
_thread_local = threading.local()


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.espn.com/",
    "Origin": "https://www.espn.com",
    "Connection": "keep-alive",
}


def get_session(pool_size=32):
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update(BROWSER_HEADERS)
        # 403 included here: on this API a 403 is often a transient
        # WAF/edge block rather than a hard ban, so it's worth retrying
        # with backoff same as a 429/5xx.
        retry = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[403, 429, 500, 502, 503, 504],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        _thread_local.session = s
    return _thread_local.session


def fetch_json(url, pool_size=32):
    _rate_limiter.acquire()
    session = get_session(pool_size)
    resp = session.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def roster_url(year, team_abbr, page=1):
    return f"{BASE}/seasons/{year}/teams/{team_abbr}/athletes?lang=en&page={page}"


def athlete_stats_url(year, athlete_id):
    return f"{BASE}/seasons/{year}/types/2/athletes/{athlete_id}/statistics?lang=en"


def athlete_id_from_ref(ref):
    m = ATHLETE_ID_PATTERN.search(ref)
    return m.group(1) if m else None


def _row_get(row, *names):
    for name in names:
        if name in row:
            value = row[name]
            if value is not None and value != "":
                return value
    return None


def fetch_roster_entries_nfl_data_py(year, team_abbr, verbose=True):
    if nfl_data_py is None:
        raise RuntimeError("nfl_data_py is not installed")

    if hasattr(nfl_data_py, "import_rosters"):
        roster_df = nfl_data_py.import_rosters([year])
    elif hasattr(nfl_data_py, "import_roster"):
        roster_df = nfl_data_py.import_roster(year)
    else:
        raise RuntimeError("nfl_data_py does not expose a roster import function")

    if roster_df is None:
        return []

    team_key = None
    for candidate in ("team_abbr", "team", "team_abbreviation", "abbr", "team_abbreviation", "teamAbbr"):
        if candidate in roster_df.columns:
            team_key = candidate
            break

    filtered = roster_df if team_key is None else roster_df[roster_df[team_key].astype(str).str.upper() == team_abbr.upper()]

    entries = []
    for _, row in filtered.iterrows():
        row_dict = row.to_dict()
        name = _row_get(row_dict, "display_name", "full_name", "name", "player_name", "displayName", "fullName")
        if name is None:
            first = _row_get(row_dict, "first_name", "firstName")
            last = _row_get(row_dict, "last_name", "lastName")
            if first or last:
                name = f"{first or ''} {last or ''}".strip()
        if name is None:
            name = "Unknown Player"

        position = _row_get(row_dict, "position", "position_abbr", "position_abbreviation", "positionAbbr")
        if not position and "position_group" in row_dict:
            position = row_dict["position_group"]

        entry = {
            "id": _row_get(row_dict, "player_id", "athlete_id", "id"),
            "displayName": name,
            "fullName": name,
            "name": name,
            "position": {"abbreviation": position} if position else {},
            "team": {"abbreviation": team_abbr},
            "debutYear": _row_get(row_dict, "debut_year", "debutYear", "draft_year", "draftYear"),
        }
        entries.append(entry)

    if verbose:
        print(f"    roster has {len(entries)} players via nfl_data_py", file=sys.stderr)
    return entries


def fetch_roster_entries(year, team_abbr, verbose=True, source="auto"):
    if source == "auto":
        source = "nfl_data_py" if nfl_data_py is not None else "espn"

    if source == "nfl_data_py":
        try:
            return fetch_roster_entries_nfl_data_py(year, team_abbr, verbose=verbose)
        except Exception as err:
            if verbose:
                print(f"    [!] nfl_data_py fallback failed: {err}; falling back to ESPN", file=sys.stderr)
            source = "espn"

    entries = []
    page = 1
    while True:
        data = fetch_json(roster_url(year, team_abbr, page=page))
        items = data.get("items", [])
        entries.extend(items)

        count = data.get("count") or len(items)
        page_size = data.get("pageSize") or len(items) or 1
        page_count = data.get("pageCount")
        if page_count is None:
            page_count = max(1, math.ceil(count / page_size)) if count else 1
        page_count = int(page_count)

        if verbose and page == 1:
            print(f"    roster has {count} players across {page_count} page(s)", file=sys.stderr)
        if page >= page_count:
            break
        page += 1
    return entries


def resolve_athlete_from_entry(entry, cache, cache_lock):
    ref = entry.get("$ref") if isinstance(entry, dict) else None

    if ref:
        athlete_id = athlete_id_from_ref(ref)
        with cache_lock:
            if athlete_id and athlete_id in cache:
                return athlete_id, cache[athlete_id]
        data = fetch_json(ref)
    else:
        data = entry
        raw_id = data.get("id")
        athlete_id = str(raw_id) if raw_id is not None else None
        with cache_lock:
            if athlete_id and athlete_id in cache:
                return athlete_id, cache[athlete_id]

    name = data.get("displayName") or data.get("fullName") or "Unknown Player"

    position_field = data.get("position") or {}
    if "abbreviation" in position_field:
        position = position_field["abbreviation"]
    elif "$ref" in position_field:
        try:
            position = fetch_json(position_field["$ref"]).get("abbreviation", "")
        except Exception:
            position = ""
    else:
        position = ""

    if not athlete_id:
        athlete_id = str(data.get("id") or name)

    debut_year = data.get("debutYear")
    if debut_year is None:
        debut_year = (data.get("draft") or {}).get("year")

    info = {"name": name, "position": position, "debut_year": debut_year}
    with cache_lock:
        cache[athlete_id] = info
    return athlete_id, info


def get_stat_value(stats_json, stat_name):
    splits = stats_json.get("splits") or {}
    for cat in splits.get("categories", []):
        for stat in cat.get("stats", []):
            if stat.get("name") == stat_name:
                value = stat.get("value")
                if value is None:
                    continue
                if isinstance(value, float) and value.is_integer():
                    value = int(value)
                return value
    return None


def build_stats_dict(stats_json, wanted_keys):
    out = {}
    for espn_name, short_key in STAT_KEY_MAP.items():
        if short_key not in wanted_keys:
            continue
        value = get_stat_value(stats_json, espn_name)
        out[short_key] = value if value is not None else 0
    return out


def _process_entry(entry, year, team_abbr, team_str, athlete_cache, cache_lock, verbose):
    """Runs on a worker thread: resolve one roster entry -> maybe fetch stats."""
    try:
        athlete_id, info = resolve_athlete_from_entry(entry, athlete_cache, cache_lock)
    except Exception as err:
        if verbose:
            print(f"    [!] couldn't resolve athlete {entry}: {err}", file=sys.stderr)
        return None

    group = POSITION_GROUP_MAP.get(info["position"])
    if not group:
        return None

    debut_year = info.get("debut_year")
    if debut_year and year < debut_year:
        if verbose:
            print(f"    - [{group}] {info['name']}: not in the league until {debut_year}, "
                  f"skipping for {year} (stale roster entry)", file=sys.stderr)
        return None

    try:
        stats_json = fetch_json(athlete_stats_url(year, athlete_id))
    except Exception as err:
        if verbose:
            print(f"    [!] couldn't fetch stats for {info['name']} ({athlete_id}): {err}", file=sys.stderr)
        return None

    stats = build_stats_dict(stats_json, WANTED_KEYS_BY_GROUP[group])

    if all(v == 0 for v in stats.values()):
        if verbose:
            print(f"    - [{group}] {info['name']} ({team_str}): all-zero stats, skipping", file=sys.stderr)
        return None

    if verbose:
        print(f"    + [{group}] {info['name']} ({team_str}): {stats}", file=sys.stderr)

    return group, {"name": info["name"], "team": team_str, "stats": stats}


def compile_team_year(year, team_abbr, verbose=True, workers=DEFAULT_WORKERS, source="auto"):
    yy = str(year)[-2:]
    team_abbr = team_abbr.upper()
    team_str = f"{team_abbr} '{yy}"
    result = {"QB": [], "RB": [], "WR": [], "TE": []}
    athlete_cache = {}
    cache_lock = threading.Lock()

    if verbose:
        if source == "nfl_data_py":
            print(f"    GET {team_abbr} roster via nfl_data_py for {year}", file=sys.stderr)
        else:
            print(f"    GET {roster_url(year, team_abbr)}", file=sys.stderr)

    try:
        entries = fetch_roster_entries(year, team_abbr, verbose=verbose, source=source)
    except Exception as err:
        print(f"    [!] couldn't fetch roster for {team_abbr} {year}: {err}", file=sys.stderr)
        return result

    # This is the part that used to be a plain sequential for-loop over
    # every roster entry (resolve athlete, then fetch stats, one at a time).
    # Farming it out to a thread pool is where almost all of the wall-clock
    # time gets clawed back, since each of these is a blocking network call.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_process_entry, entry, year, team_abbr, team_str, athlete_cache, cache_lock, verbose)
            for entry in entries
        ]
        for future in as_completed(futures):
            outcome = future.result()
            if outcome is None:
                continue
            group, player = outcome
            result[group].append(player)

    return result


def merge_team_results(accumulator, year_result):
    for group, players in year_result.items():
        accumulator.setdefault(group, []).extend(players)


def write_team_file(team_abbr, data, out_dir="."):
    import json
    fname = f"{team_abbr.upper()}_Stats.json"
    out_path = f"{out_dir.rstrip('/')}/{fname}" if out_dir != "." else fname
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    counts = {k: len(v) for k, v in data.items()}
    print(f"  -> wrote {out_path} ({counts})", file=sys.stderr)
    return out_path


def collect_teams(teams_arg, divisions_arg):
    teams = []
    if divisions_arg:
        requested = [d.strip() for d in divisions_arg.split(",") if d.strip()]
        if any(d.lower() == "all" for d in requested):
            for div_teams in DIVISIONS.values():
                teams.extend(div_teams)
        else:
            for d in requested:
                teams.extend(teams_for_division(d))

    if teams_arg:
        teams.extend(t.strip().upper() for t in teams_arg.split(",") if t.strip())

    seen = set()
    deduped = []
    for t in teams:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Compile NFL team roster stats into blind-draft player JSON (threaded/pooled).")
    parser.add_argument("year", type=str, help="Season year (e.g. 2026), or 'all' to compile every season in --start..--end, merged per team")
    parser.add_argument("--divisions", type=str, default=None,
                         help="Comma-separated NFL divisions, e.g. 'NFC North,AFC South'. Use 'all' for every division. "
                              f"Valid names: {', '.join(DIVISIONS.keys())}")
    parser.add_argument("--teams", type=str, default=None, help="Comma-separated team abbreviations, e.g. 'GB,KC,DAL'")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output on stderr")
    parser.add_argument("--start", type=int, default=DEFAULT_ALL_START, help=f"Newest year for 'all' mode (default: {DEFAULT_ALL_START})")
    parser.add_argument("--end", type=int, default=DEFAULT_ALL_END, help=f"Oldest year for 'all' mode (default: {DEFAULT_ALL_END})")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to write '{TEAM}_Stats.json' files to")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                         help=f"Concurrent threads for athlete resolve+stats fetches (default: {DEFAULT_WORKERS}). "
                              "Higher = more requests in flight at once; 10-25 is a reasonable range.")
    parser.add_argument("--max-rps", type=int, default=DEFAULT_MAX_RPS,
                         help=f"Global cap on requests started per second, across all threads (default: {DEFAULT_MAX_RPS}). "
                              "This is the main knob for avoiding 403s -- lower it if you still get blocked.")
    parser.add_argument("--source", type=str, choices=["auto", "espn", "nfl_data_py"], default="auto",
                         help="Data source for roster fetches: 'auto' prefers nfl_data_py if installed, else ESPN.")
    args = parser.parse_args()
    global _rate_limiter
    _rate_limiter = RateLimiter(args.max_rps)

    try:
        teams = collect_teams(args.teams, args.divisions)
    except ValueError as err:
        parser.error(str(err))

    if not teams:
        parser.error("Must specify --teams and/or --divisions (or --divisions all).")

    verbose = not args.quiet

    if args.year.lower() == "all":
        lo, hi = sorted((args.start, args.end))
        years = list(range(hi, lo - 1, -1))
        print(f"Compiling {len(teams)} team(s) across {args.start}..{args.end} (workers={args.workers})...", file=sys.stderr)

        current_division = None
        for team in teams:
            division = TEAM_TO_DIVISION.get(team)
            if division and division != current_division:
                print(f"\n##### {division} #####", file=sys.stderr)
                current_division = division
            print(f"\n=== {team} ===", file=sys.stderr)

            merged = {"QB": [], "RB": [], "WR": [], "TE": []}
            for year in years:
                print(f"  -- {year} --", file=sys.stderr)
                data = compile_team_year(year, team, verbose=verbose, workers=args.workers, source=args.source)
                merge_team_results(merged, data)

            write_team_file(team, merged, out_dir=args.out_dir)
        return

    year = int(args.year)
    print(f"Compiling {len(teams)} team(s) for {year} (workers={args.workers})...", file=sys.stderr)
    current_division = None
    for team in teams:
        division = TEAM_TO_DIVISION.get(team)
        if division and division != current_division:
            print(f"\n##### {division} #####", file=sys.stderr)
            current_division = division
        print(f"\n=== {team} ===", file=sys.stderr)
        data = compile_team_year(year, team, verbose=verbose, workers=args.workers, source=args.source)
        write_team_file(team, data, out_dir=args.out_dir)


if __name__ == "__main__":
    main()