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
     -> resolve that athlete's name, position, team $ref, and college $ref.
  3. GET .../seasons/{year}/types/2/athletes/{id}/statistics?lang=en
     -> pull the actual per-category stat lines (passing/rushing/receiving)
        for that athlete's season.
  4. Assemble everything into:
        {"name": "Greg Olsen", "team": "CAR '19", "stats": {"recyds": 597, "rectd": 2, "rec": 52}}
     grouped by position (QB / RB / WR / TE), matching the shape the game's
     TWISTS.players object expects.

College filtering (new):
  Pass --colleges "LSU,OU,Alabama" (names or ESPN abbreviations, comma
  separated, case-insensitive) and the script will:
    - resolve each athlete's college via the "college" -> "$ref" field
      returned by the athlete endpoint (a GET to
      https://sports.core.api.espn.com/v2/colleges/{id}?lang=en),
    - drop any athlete whose college doesn't match one of the given
      names/abbreviations,
    - group survivors by college instead of dumping everyone into one
      pile, and write one file per college instead of one file per year:
      "LSU_Stats.json", "OU_Stats.json", etc. Each of those files has the
      same {"QB": [...], "RB": [...], "WR": [...], "TE": [...]} shape as
      before -- just scoped to that school.
  In "all" mode with --colleges set, results are accumulated across every
  season in the range and written once at the end (so LSU_Stats.json ends
  up with LSU alums from every year processed, not just the last one).

Usage:
    # single year -> prints JSON to stdout
    python espn_leaders.py 2012

    # single year -> writes to a specific file
    python espn_leaders.py 2012 --output 2012Stats.json

    # single year, filtered to specific colleges -> writes LSU_Stats.json, OU_Stats.json
    python espn_leaders.py 2012 --colleges "LSU,OU"

    # every season from 2025 down to 1980 -> writes 2025Stats.json, 2024Stats.json, ... 1980Stats.json
    python espn_leaders.py all

    # every season, filtered to specific colleges -> writes LSU_Stats.json, OU_Stats.json
    # (each containing every matching player found across the whole range)
    python espn_leaders.py all --colleges "LSU,OU,Alabama"

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
DEFAULT_ALL_END = 1999    # oldest year in the default "all" range

# ESPN leader-category name -> which position group it feeds, and which
# on-field positions are allowed to actually count for that category.
# (A QB scrambling for rushing yards still shouldn't end up in the RB pool.)
CATEGORY_CONFIG = {
    "passingYards": {"group": "QB", "positions": {"QB"}},
    "rushingYards": {"group": "RB", "positions": {"RB", "FB"}},
    "receivingTouchdowns": {"group": None, "positions": {"WR", "TE"}},  # group decided per-athlete
    "rushingTouchdowns": {"group": "RB", "positions": {"RB", "FB", "QB"}},
    "receivingYards": {"group": None, "positions": {"WR", "TE"}},
    "receptions": {"group": None, "positions": {"WR", "TE", "RB"}}  # group decided per-athlete
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

ATHLETE_ID_PATTERN = re.compile(r"/athletes/(\d+)")
COLLEGE_ID_PATTERN = re.compile(r"/colleges/(\d+)")


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
    m = ATHLETE_ID_PATTERN.search(ref)
    return m.group(1) if m else None


def college_id_from_ref(ref):
    m = COLLEGE_ID_PATTERN.search(ref)
    return m.group(1) if m else None


def get_leader_category(leaders_json, category_name):
    for cat in leaders_json.get("categories", []):
        if cat.get("name") == category_name:
            return cat
    return None


def resolve_college(college_ref, cache):
    """Fetch (and cache) a college's name/abbreviation from its $ref."""
    college_id = college_id_from_ref(college_ref)
    if college_id and college_id in cache:
        return cache[college_id]

    try:
        data = fetch_json(college_ref)
    except RuntimeError:
        return None

    info = {
        "id": college_id,
        "name": data.get("name") or data.get("displayName"),
        "abbreviation": data.get("abbreviation"),
    }
    if college_id:
        cache[college_id] = info
    return info


def resolve_athlete(year, athlete_id, athlete_cache, college_cache):
    """Fetch (and cache) an athlete's name, position, team, and college."""
    if athlete_id in athlete_cache:
        return athlete_cache[athlete_id]

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

    college_abbr = None
    college_name = None
    college_ref = (data.get("college") or {}).get("$ref")
    if college_ref:
        college_info = resolve_college(college_ref, college_cache)
        if college_info:
            college_abbr = college_info.get("abbreviation")
            college_name = college_info.get("name")

    info = {
        "name": name,
        "position": position,
        "team_abbr": team_abbr,
        "college_abbr": college_abbr,
        "college_name": college_name,
    }
    athlete_cache[athlete_id] = info
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


def normalize_college_token(s):
    return (s or "").strip().upper()


def matches_target_colleges(info, target_set):
    """True if the athlete's college abbreviation or name is in target_set."""
    if not target_set:
        return True
    candidates = {
        normalize_college_token(info.get("college_abbr")),
        normalize_college_token(info.get("college_name")),
    }
    candidates.discard("")
    return bool(candidates & target_set)


def sanitize_filename(s):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", s or "").strip("_")
    return cleaned or "UNKNOWN"


def compile_year(year, top_n=25, verbose=True, target_colleges=None):
    """
    Returns player data for one season.

    If target_colleges is falsy: same shape as before --
        {"QB": [...], "RB": [...], "WR": [...], "TE": [...]}

    If target_colleges is a set of upper-cased college names/abbreviations,
    only players whose college matches one of them are kept, and the result
    is grouped by college first:
        {"LSU": {"QB": [...], "RB": [...], "WR": [...], "TE": [...]},
         "OU":  {"QB": [...], "RB": [...], "WR": [...], "TE": [...]}}
    """
    yy = str(year)[-2:]
    leaders_json = fetch_json(leaders_url(year))
    athlete_cache = {}
    college_cache = {}

    if target_colleges:
        result = {}
    else:
        result = {"QB": [], "RB": [], "WR": [], "TE": []}

    seen = set()  # (college_key_or_None, group, athlete_id)

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

            info = resolve_athlete(year, athlete_id, athlete_cache, college_cache)
            position = info["position"]
            if position not in allowed_positions:
                continue  # e.g. a QB who shows up on the rushing leaderboard

            if target_colleges and not matches_target_colleges(info, target_colleges):
                continue

            group = config["group"] or position  # receivingYards splits by actual position
            if group not in {"QB", "RB", "WR", "TE"}:
                continue

            college_key = None
            if target_colleges:
                college_key = info.get("college_abbr") or info.get("college_name") or "UNKNOWN"

            dedup_key = (college_key, group, athlete_id)
            if dedup_key in seen:
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
            player_entry = {"name": info["name"], "team": team_str, "stats": stats}

            if target_colleges:
                bucket = result.setdefault(college_key, {"QB": [], "RB": [], "WR": [], "TE": []})
                bucket[group].append(player_entry)
            else:
                result[group].append(player_entry)

            seen.add(dedup_key)

            if verbose:
                tag = f"{college_key} " if target_colleges else ""
                print(f"  + [{tag}{group}] {info['name']} ({team_str}): {stats}", file=sys.stderr)

    return result


def merge_college_results(accumulator, year_result):
    """Merge one season's college-grouped result into a running accumulator."""
    for college_key, groups in year_result.items():
        bucket = accumulator.setdefault(college_key, {"QB": [], "RB": [], "WR": [], "TE": []})
        for group, players in groups.items():
            bucket[group].extend(players)


def write_college_files(college_data, out_dir="."):
    """Write one "{College}_Stats.json" file per college in college_data."""
    written = []
    for college_key, groups in college_data.items():
        fname = f"{sanitize_filename(college_key)}_Stats.json"
        out_path = f"{out_dir.rstrip('/')}/{fname}" if out_dir != "." else fname
        with open(out_path, "w") as f:
            json.dump(groups, f, indent=2)
        counts = {k: len(v) for k, v in groups.items()}
        print(f"  -> wrote {out_path} ({counts})", file=sys.stderr)
        written.append(out_path)
    return written


def compile_all_years(start_year, end_year, top_n=25, verbose=True, out_dir=".", target_colleges=None):
    """
    Compiles every season from start_year down to end_year (inclusive,
    regardless of which is larger).

    - Without target_colleges: writes "{year}Stats.json" per season, same as
      before. Returns a list of (year, path, error) tuples.
    - With target_colleges: accumulates matching players across every season
      in the range, then writes one "{College}_Stats.json" per college at the
      end. Returns a list of (year, None, error) tuples (one per season
      processed) so the caller can still see which years failed.
    """
    lo, hi = sorted((start_year, end_year))
    years = list(range(hi, lo - 1, -1))  # newest -> oldest, matches "2025Stats.json etc"

    if target_colleges:
        merged = {}
        results = []
        for year in years:
            print(f"\n=== {year} ===", file=sys.stderr)
            try:
                data = compile_year(year, top_n=top_n, verbose=verbose, target_colleges=target_colleges)
                merge_college_results(merged, data)
                results.append((year, None, None))
            except Exception as err:
                print(f"  [!] failed for {year}: {err}", file=sys.stderr)
                results.append((year, None, str(err)))

        print(file=sys.stderr)
        written = write_college_files(merged, out_dir=out_dir)

        ok = [r for r in results if r[2] is None]
        failed = [r for r in results if r[2] is not None]
        print(f"\nDone: {len(ok)} seasons processed, {len(failed)} failed. Wrote {len(written)} college file(s).", file=sys.stderr)
        if failed:
            print("Failed years:", ", ".join(str(y) for y, _, _ in failed), file=sys.stderr)
        return results

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
    parser.add_argument("--top", type=int, default=50, help="How many leaders to pull per category (default: 25)")
    parser.add_argument("--output", type=str, default=None, help="Optional file path to write a single year's JSON to (ignored if --colleges is set)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output on stderr")
    parser.add_argument("--start", type=int, default=DEFAULT_ALL_START, help=f"Newest year for 'all' mode (default: {DEFAULT_ALL_START})")
    parser.add_argument("--end", type=int, default=DEFAULT_ALL_END, help=f"Oldest year for 'all' mode (default: {DEFAULT_ALL_END})")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to write output files to (default: current directory)")
    parser.add_argument(
        "--colleges",
        type=str,
        default=None,
        help="Comma-separated college names or ESPN abbreviations to filter by, e.g. 'LSU,OU,Alabama'. "
             "Case-insensitive. When set, output is grouped and written per college as "
             "'{College}_Stats.json' instead of per year.",
    )
    args = parser.parse_args()

    target_colleges = None
    if args.colleges:
        target_colleges = {normalize_college_token(c) for c in args.colleges.split(",") if c.strip()}

    if args.year.lower() == "all":
        print(f"Compiling every season {args.start} down to {args.end} (top {args.top} per category)...", file=sys.stderr)
        if target_colleges:
            print(f"Filtering to colleges: {', '.join(sorted(target_colleges))}", file=sys.stderr)
        compile_all_years(
            args.start, args.end, top_n=args.top, verbose=not args.quiet,
            out_dir=args.out_dir, target_colleges=target_colleges,
        )
        return

    year = int(args.year)
    print(f"Compiling {year} leaders (top {args.top} per category)...", file=sys.stderr)
    if target_colleges:
        print(f"Filtering to colleges: {', '.join(sorted(target_colleges))}", file=sys.stderr)

    result = compile_year(year, top_n=args.top, verbose=not args.quiet, target_colleges=target_colleges)

    if target_colleges:
        write_college_files(result, out_dir=args.out_dir)
        return

    output_json = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output_json)
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()