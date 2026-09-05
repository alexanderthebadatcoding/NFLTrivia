#!/usr/bin/env python3
"""
espn_leaders.py

Pull a season's statistical leaders from ESPN's public core API and compile
them into the {name, team, stats} player format used by the Blind Draft
Challenge game.

Flow, per year:
  1. GET .../seasons/{year}/types/2/leaders?lang=en
     -> for each stat category we care about (passing yards, rushing yards,
        receiving yards), grab the top N leader entries. Each entry only has
        a value/displayValue plus a $ref to the athlete.
  2. GET .../seasons/{year}/athletes/{id}?lang=en
     -> resolve that athlete's name, position, and team $ref.
  3. GET .../seasons/{year}/types/2/athletes/{id}/statistics?lang=en
     -> pull the actual per-category stat lines (passing/rushing/receiving)
        for that athlete's season.
  4. Assemble everything into:
        {"name": "Greg Olsen", "team": "CAR '19", "stats": {"recyds": 597, "rectd": 2, "rec": 52}}
     grouped by position (QB / RB / WR / TE), matching the shape the game's
     TWISTS.players object expects.

Usage:
    # single year -> prints JSON to stdout
    python espn_leaders.py 2012

    # single year -> writes to a specific file
    python espn_leaders.py 2012 --output 2012Stats.json

    # every season from 2025 down to 1980 -> writes 2025Stats.json, 2024Stats.json, ... 1980Stats.json
    python espn_leaders.py all

    # a custom range, same "all" behavior
    python espn_leaders.py all --start 2010 --end 2000
"""

import argparse
import json
import re
import sys
import time
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

BASE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"

DEFAULT_ALL_START = 2025  # newest year in the default "all" range
DEFAULT_ALL_END = 1980    # oldest year in the default "all" range

# ESPN leader-category name -> which position group it feeds, and which
# on-field positions are allowed to actually count for that category.
# (A QB scrambling for rushing yards still shouldn't end up in the RB pool.)
CATEGORY_CONFIG = {
    "passingYards": {"group": "QB", "positions": {"QB"}},
    "rushingYards": {"group": "RB", "positions": {"RB", "FB"}},
    "receivingYards": {"group": None, "positions": {"WR", "TE"}},  # group decided per-athlete
}

# Stat name (as it appears inside statistics -> splits -> categories -> stats)
# -> the short key used in the game's stats object.
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

REQUEST_DELAY = 0.05  # be polite to ESPN's API
TIMEOUT = 15


def fetch_json(url, retries=3):
    """GET a URL and parse it as JSON, with a couple of retries on failure."""
    last_err = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (blind-draft-challenge script)"})
            with urlopen(req, timeout=TIMEOUT) as resp:
                data = json.load(resp)
            time.sleep(REQUEST_DELAY)
            return data
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as err:
            last_err = err
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def leaders_url(year):
    return f"{BASE}/seasons/{year}/types/2/leaders?lang=en"


def athlete_url(year, athlete_id):
    return f"{BASE}/seasons/{year}/athletes/{athlete_id}?lang=en"


def athlete_stats_url(year, athlete_id):
    return f"{BASE}/seasons/{year}/types/2/athletes/{athlete_id}/statistics?lang=en"


def athlete_id_from_ref(ref):
    m = re.search(r"/athletes/(\d+)\?", ref)
    return m.group(1) if m else None


def get_leader_category(leaders_json, category_name):
    for cat in leaders_json.get("categories", []):
        if cat.get("name") == category_name:
            return cat
    return None


def resolve_athlete(year, athlete_id, cache):
    """Fetch (and cache) an athlete's name, position, and team abbreviation."""
    if athlete_id in cache:
        return cache[athlete_id]

    data = fetch_json(athlete_url(year, athlete_id))
    name = data.get("displayName") or data.get("fullName") or "Unknown Player"

    # position sometimes comes back embedded, sometimes as a $ref that needs
    # its own fetch
    position_field = data.get("position") or {}
    if "abbreviation" in position_field:
        position = position_field["abbreviation"]
    elif "$ref" in position_field:
        try:
            position = fetch_json(position_field["$ref"]).get("abbreviation", "")
        except RuntimeError:
            position = ""
    else:
        position = ""

    team_abbr = None
    team_ref = (data.get("team") or {}).get("$ref")
    if team_ref:
        try:
            team_data = fetch_json(team_ref)
            team_abbr = team_data.get("abbreviation")
        except RuntimeError:
            team_abbr = None

    info = {"name": name, "position": position, "team_abbr": team_abbr}
    cache[athlete_id] = info
    return info


def get_stat_value(stats_json, stat_name):
    """Search every category in the statistics response for a stat by name."""
    splits = stats_json.get("splits") or {}
    for cat in splits.get("categories", []):
        for stat in cat.get("stats", []):
            if stat.get("name") == stat_name:
                value = stat.get("value")
                if value is None:
                    continue
                # ESPN often returns whole-number stats as floats (e.g. 597.0)
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


def compile_year(year, top_n=25, verbose=True):
    """
    Returns a dict shaped like:
        {"QB": [...], "RB": [...], "WR": [...], "TE": [...]}
    where each entry is {"name": ..., "team": ..., "stats": {...}}.
    """
    yy = str(year)[-2:]
    leaders_json = fetch_json(leaders_url(year))
    athlete_cache = {}
    result = {"QB": [], "RB": [], "WR": [], "TE": []}
    seen_per_group = {"QB": set(), "RB": set(), "WR": set(), "TE": set()}

    for category_name, config in CATEGORY_CONFIG.items():
        cat = get_leader_category(leaders_json, category_name)
        if not cat or not cat.get("leaders"):
            if verbose:
                print(f"  [!] no '{category_name}' leaders for {year} — skipping", file=sys.stderr)
            continue

        allowed_positions = config["positions"]
        entries = cat["leaders"][:top_n]

        for entry in entries:
            athlete_ref = (entry.get("athlete") or {}).get("$ref")
            athlete_id = athlete_id_from_ref(athlete_ref) if athlete_ref else None
            if not athlete_id:
                continue

            info = resolve_athlete(year, athlete_id, athlete_cache)
            position = info["position"]
            if position not in allowed_positions:
                continue  # e.g. a QB who shows up on the rushing leaderboard

            group = config["group"] or position  # receivingYards splits by actual position
            if group not in result:
                continue
            if athlete_id in seen_per_group[group]:
                continue  # already added to this group from another category

            try:
                stats_json = fetch_json(athlete_stats_url(year, athlete_id))
            except RuntimeError as err:
                if verbose:
                    print(f"  [!] couldn't fetch stats for {info['name']} ({athlete_id}): {err}", file=sys.stderr)
                continue

            wanted_keys = {
                "QB": {"yds", "td", "int"},
                "RB": {"ryds", "rtd", "recyds", "rectd", "rec"},
                "WR": {"recyds", "rectd", "rec"},
                "TE": {"recyds", "rectd", "rec"},
            }[group]

            stats = build_stats_dict(stats_json, wanted_keys)
            team_str = f"{info['team_abbr']} '{yy}" if info["team_abbr"] else f"NFL '{yy}"

            result[group].append({"name": info["name"], "team": team_str, "stats": stats})
            seen_per_group[group].add(athlete_id)

            if verbose:
                print(f"  + [{group}] {info['name']} ({team_str}): {stats}", file=sys.stderr)

    return result


def compile_all_years(start_year, end_year, top_n=25, verbose=True, out_dir="."):
    """
    Compiles every season from start_year down to end_year (inclusive,
    regardless of which is larger) and writes each one to
    "{year}Stats.json" in out_dir. Returns a list of (year, path, error)
    tuples so the caller can see what succeeded/failed.
    """
    lo, hi = sorted((start_year, end_year))
    years = list(range(hi, lo - 1, -1))  # newest -> oldest, matches "2025Stats.json etc"

    results = []
    for year in years:
        out_path = f"{out_dir.rstrip('/')}/{year}Stats.json" if out_dir != "." else f"{year}Stats.json"
        print(f"\n=== {year} ===", file=sys.stderr)
        try:
            data = compile_year(year, top_n=top_n, verbose=verbose)
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            counts = {k: len(v) for k, v in data.items()}
            print(f"  -> wrote {out_path} ({counts})", file=sys.stderr)
            results.append((year, out_path, None))
        except Exception as err:
            print(f"  [!] failed for {year}: {err}", file=sys.stderr)
            results.append((year, out_path, str(err)))
    return results


def main():
    parser = argparse.ArgumentParser(description="Compile ESPN season stat leaders into blind-draft player JSON.")
    parser.add_argument(
        "year",
        type=str,
        help="Season year (e.g. 2012), or 'all' to compile every season in --start..--end "
             "and save each as '{year}Stats.json'",
    )
    parser.add_argument("--top", type=int, default=25, help="How many leaders to pull per category (default: 25)")
    parser.add_argument("--output", type=str, default=None, help="Optional file path to write a single year's JSON to")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output on stderr")
    parser.add_argument("--start", type=int, default=DEFAULT_ALL_START, help=f"Newest year for 'all' mode (default: {DEFAULT_ALL_START})")
    parser.add_argument("--end", type=int, default=DEFAULT_ALL_END, help=f"Oldest year for 'all' mode (default: {DEFAULT_ALL_END})")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to write '{year}Stats.json' files to in 'all' mode (default: current directory)")
    args = parser.parse_args()

    if args.year.lower() == "all":
        print(f"Compiling every season {args.start} down to {args.end} (top {args.top} per category)...", file=sys.stderr)
        results = compile_all_years(args.start, args.end, top_n=args.top, verbose=not args.quiet, out_dir=args.out_dir)
        ok = [r for r in results if r[2] is None]
        failed = [r for r in results if r[2] is not None]
        print(f"\nDone: {len(ok)} seasons written, {len(failed)} failed.", file=sys.stderr)
        if failed:
            print("Failed years:", ", ".join(str(y) for y, _, _ in failed), file=sys.stderr)
        return

    year = int(args.year)
    print(f"Compiling {year} leaders (top {args.top} per category)...", file=sys.stderr)
    result = compile_year(year, top_n=args.top, verbose=not args.quiet)

    output_json = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()