#!/usr/bin/env python3
"""
nfl_roster_stats.py

Rewrite of espn_roster_stats_fast.py that pulls from nflverse (via the
nflreadpy package) instead of ESPN's undocumented API.

Why this is simpler than the ESPN version:
    nflverse doesn't expose a per-athlete REST endpoint. It publishes whole
    tables (one big parquet/csv per data type, covering every team and
    player) that you download once per season and then filter/group in
    memory. That means no requests.Session pooling, no rate limiter, no
    ThreadPoolExecutor, no per-player HTTP round trip, and no 403 handling --
    one load_rosters() call and one load_player_stats() call per season
    covers every team you ask for.

Output format matches the original script's per-team JSON files, but now one
file per team *per season* instead of merged across years, e.g.:

    data/GB2025.json
    data/GB2024.json
    data/KC2025.json

Each file looks like:
    {
      "QB": [{"name": ..., "team": "GB '25", "stats": {"yds":..,"td":..,"int":..}}],
      "RB": [{"name": ..., "team": "GB '25", "stats": {"ryds":..,"rtd":..,"recyds":..,"rectd":..,"rec":..}}],
      "WR": [...],
      "TE": [...]
    }

Install:
    pip install nflreadpy --break-system-packages

Usage (mirrors the original script):
    python nfl_roster_stats.py 2025 --divisions "NFC North"
    python nfl_roster_stats.py 2025 --teams GB,KC,DAL
    python nfl_roster_stats.py all --start 2025 --end 2016 --divisions all
"""

import argparse
import json
import re
import sys

import nflreadpy as nfl
import polars as pl

# ---------------------------------------------------------------------------
# Divisions / abbreviations. Kept in the ESPN-style codes you originally used
# on the command line (--teams GB,KC,...); we translate to whatever code
# nflverse actually uses for a given season right before filtering, since
# nflverse's team codes have changed historically for a few franchises
# (e.g. the Rams/Washington/Raiders have gone by more than one abbreviation
# across nflverse's data history). See resolve_team_code() below.
# ---------------------------------------------------------------------------
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

# Known alternate spellings nflverse's data has used for a team code. We try
# each alias against whatever team codes actually show up in that season's
# roster table, so this stays correct even if nflverse changes conventions.
ABBR_ALIASES = {
    "WSH": ["WSH", "WAS"],
    "LAR": ["LAR", "LA"],
    "JAX": ["JAX", "JAC"],
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


def resolve_team_code(requested_abbr, known_codes):
    """Map our ESPN-style abbreviation to whatever code nflverse used that season."""
    candidates = ABBR_ALIASES.get(requested_abbr, [requested_abbr])
    for candidate in candidates:
        if candidate in known_codes:
            return candidate
    return None


# ---------------------------------------------------------------------------
# Position grouping + stat key mapping (same shape as the original script)
# ---------------------------------------------------------------------------
POSITION_GROUP_MAP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}

WANTED_KEYS_BY_GROUP = {
    "QB": {"yds", "td", "int"},
    "RB": {"ryds", "rtd", "recyds", "rectd", "rec"},
    "WR": {"recyds", "rectd", "rec"},
    "TE": {"recyds", "rectd", "rec"},
}

# nflverse column name -> our short stat key
STAT_COL_MAP = {
    "passing_yards": "yds",
    "passing_tds": "td",
    "passing_interceptions": "int",
    "rushing_yards": "ryds",
    "rushing_tds": "rtd",
    "receiving_yards": "recyds",
    "receiving_tds": "rectd",
    "receptions": "rec",
}


# ---------------------------------------------------------------------------
# Data loading (one call per season, reused across every team in that season)
# ---------------------------------------------------------------------------
_roster_cache = {}
_stats_cache = {}


def load_season_rosters(year, verbose=True):
    if year not in _roster_cache:
        if verbose:
            print(f"  loading {year} rosters via nflreadpy.load_rosters()...", file=sys.stderr)
        _roster_cache[year] = nfl.load_rosters(seasons=[year])
    return _roster_cache[year]


def load_season_stats(year, verbose=True):
    if year not in _stats_cache:
        if verbose:
            print(f"  loading {year} player stats via nflreadpy.load_player_stats()...", file=sys.stderr)
        # summary_level="reg" aggregates each player's regular-season games into
        # one row per player per season (the analogue of ESPN's types/2 stats).
        df = nfl.load_player_stats(seasons=[year], summary_level="reg")
        _stats_cache[year] = df
    return _stats_cache[year]


def compile_team_season(year, team_abbr, verbose=True):
    yy = str(year)[-2:]
    team_abbr = team_abbr.upper()
    result = {"QB": [], "RB": [], "WR": [], "TE": []}

    rosters = load_season_rosters(year, verbose=verbose)
    stats = load_season_stats(year, verbose=verbose)

    known_codes = set(rosters["team"].unique().to_list())
    resolved_code = resolve_team_code(team_abbr, known_codes)
    if resolved_code is None:
        print(f"    [!] no roster data found for '{team_abbr}' in {year} "
              f"(known codes this season include: {sorted(known_codes)[:6]}...)", file=sys.stderr)
        return result

    team_str = f"{team_abbr} '{yy}"

    team_roster = rosters.filter(pl.col("team") == resolved_code)
    # position lookup keyed by gsis id, since load_player_stats() also carries
    # its own 'position' column but the roster table is the more reliable source
    position_by_id = dict(zip(team_roster["gsis_id"].to_list(), team_roster["position"].to_list()))

    id_col = "player_id" if "player_id" in stats.columns else "gsis_id"
    team_col = "recent_team" if "recent_team" in stats.columns else "team"
    team_stats = stats.filter(pl.col(team_col) == resolved_code)

    for row in team_stats.iter_rows(named=True):
        player_id = row.get(id_col)
        position = position_by_id.get(player_id) or row.get("position")
        group = POSITION_GROUP_MAP.get(position)
        if not group:
            continue

        wanted = WANTED_KEYS_BY_GROUP[group]
        player_stats = {}
        for col, short_key in STAT_COL_MAP.items():
            if short_key not in wanted:
                continue
            value = row.get(col)
            if value is None:
                value = 0
            elif isinstance(value, float) and value.is_integer():
                value = int(value)
            player_stats[short_key] = value

        if all(v == 0 for v in player_stats.values()):
            continue

        name = row.get("player_display_name") or row.get("player_name") or "Unknown Player"

        if verbose:
            print(f"    + [{group}] {name} ({team_str}): {player_stats}", file=sys.stderr)

        result[group].append({"name": name, "team": team_str, "stats": player_stats})

    return result


GROUP_POSITIONS = {"QB": ["QB"], "RB": ["RB", "FB"], "WR": ["WR"], "TE": ["TE"]}


def _row_to_entry(row, group, team_col, yy):
    wanted = WANTED_KEYS_BY_GROUP[group]
    player_stats = {}
    for col, short_key in STAT_COL_MAP.items():
        if short_key not in wanted:
            continue
        value = row.get(col)
        if value is None:
            value = 0
        elif isinstance(value, float) and value.is_integer():
            value = int(value)
        player_stats[short_key] = value

    name = row.get("player_display_name") or row.get("player_name") or "Unknown Player"
    team_abbr = row.get(team_col) or "FA"
    team_str = f"{team_abbr} '{yy}"
    return {"name": name, "team": team_str, "stats": player_stats}


def compile_league_top(year, top_n=35, per_position=False, verbose=True):
    """
    Leaguewide top skill players for a season, ranked by PPR fantasy points,
    regardless of which --teams/--divisions were requested -- this is meant
    as a standalone draft-pool file (e.g. 1999Stats.json), not filtered to a
    subset of teams.

    per_position=False (default): a single pool of the top `top_n` players
      overall, split into QB/RB/WR/TE groups by how they happen to rank.
    per_position=True: the top `top_n` players *within each* of QB/RB/WR/TE,
      ranked separately -- e.g. --top-n 25 --per-position gives you 25 QBs,
      25 RBs, 25 WRs, and 25 TEs.
    """
    result = {"QB": [], "RB": [], "WR": [], "TE": []}
    stats = load_season_stats(year, verbose=verbose)

    if "fantasy_points_ppr" not in stats.columns:
        print(f"    [!] no fantasy_points_ppr column for {year}, skipping league top-{top_n}", file=sys.stderr)
        return result

    team_col = "recent_team" if "recent_team" in stats.columns else "team"
    yy = str(year)[-2:]

    if per_position:
        for group, positions in GROUP_POSITIONS.items():
            pool = stats.filter(pl.col("position").is_in(positions))
            pool = pool.sort("fantasy_points_ppr", descending=True).head(top_n)
            for row in pool.iter_rows(named=True):
                entry = _row_to_entry(row, group, team_col, yy)
                if verbose:
                    print(f"    + [{group}] {entry['name']} ({entry['team']}): {entry['stats']} "
                          f"[{row.get('fantasy_points_ppr'):.1f} pts]", file=sys.stderr)
                result[group].append(entry)
        return result

    skill = stats.filter(pl.col("position").is_in(list(POSITION_GROUP_MAP.keys())))
    skill = skill.sort("fantasy_points_ppr", descending=True).head(top_n)

    for row in skill.iter_rows(named=True):
        group = POSITION_GROUP_MAP.get(row.get("position"))
        if not group:
            continue
        entry = _row_to_entry(row, group, team_col, yy)
        if verbose:
            print(f"    + [{group}] {entry['name']} ({entry['team']}): {entry['stats']} "
                  f"[{row.get('fantasy_points_ppr'):.1f} pts]", file=sys.stderr)
        result[group].append(entry)

    return result


def write_league_year_file(year, data, out_dir="data"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{year}Stats.json"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    counts = {k: len(v) for k, v in data.items()}
    print(f"  -> wrote {out_path} ({counts})", file=sys.stderr)
    return out_path


def write_team_year_file(team_abbr, year, data, out_dir="data"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{team_abbr.upper()}{year}.json"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    counts = {k: len(v) for k, v in data.items()}
    print(f"  -> wrote {out_path} ({counts})", file=sys.stderr)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Compile NFL team roster stats into blind-draft player JSON, one file per team per season, via nflreadpy."
    )
    parser.add_argument("year", type=str, help="Season year (e.g. 2025), or 'all' to compile every season in --start..--end")
    parser.add_argument("--divisions", type=str, default=None,
                         help="Comma-separated NFL divisions, e.g. 'NFC North,AFC South'. Use 'all' for every division. "
                              f"Valid names: {', '.join(DIVISIONS.keys())}")
    parser.add_argument("--teams", type=str, default=None, help="Comma-separated team abbreviations, e.g. 'GB,KC,DAL'")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output on stderr")
    parser.add_argument("--start", type=int, default=2025, help="Newest year for 'all' mode (default: 2025)")
    parser.add_argument("--end", type=int, default=2016, help="Oldest year for 'all' mode (default: 2016)")
    parser.add_argument("--out-dir", type=str, default="data", help="Directory to write output files to (default: data)")
    parser.add_argument("--top-n", type=int, default=None, metavar="N",
                         help="Also write a leaguewide '{YEAR}Stats.json'. By default N is the size of a "
                              "single overall pool (top N skill players by PPR fantasy points, across all "
                              "positions and all 32 teams). With --per-position, N applies separately to "
                              "each of QB/RB/WR/TE (e.g. --top-n 25 --per-position gives 25 of each).")
    parser.add_argument("--per-position", action="store_true",
                         help="Interpret --top-n as a per-position count (top N QBs, N RBs, N WRs, N TEs) "
                              "instead of one combined pool of N players overall.")
    parser.add_argument("--top-n-only", action="store_true",
                         help="Skip per-team '{TEAM}{YEAR}.json' files entirely and only write the "
                              "leaguewide '{YEAR}Stats.json' from --top-n. Requires --top-n.")
    args = parser.parse_args()

    if args.top_n_only and args.top_n is None:
        parser.error("--top-n-only requires --top-n.")

    teams = []
    if not args.top_n_only:
        try:
            teams = collect_teams(args.teams, args.divisions)
        except ValueError as err:
            parser.error(str(err))

        if not teams:
            parser.error("Must specify --teams and/or --divisions (or --divisions all), unless using --top-n-only.")

    verbose = not args.quiet

    if args.year.lower() == "all":
        lo, hi = sorted((args.start, args.end))
        years = list(range(hi, lo - 1, -1))
    else:
        years = [int(args.year)]

    if teams:
        print(f"Compiling {len(teams)} team(s) across {years}...", file=sys.stderr)
    if args.top_n:
        print(f"Also compiling leaguewide top-{args.top_n} for {years}...", file=sys.stderr)

    for year in years:
        print(f"\n=== {year} ===", file=sys.stderr)
        current_division = None
        for team in teams:
            division = TEAM_TO_DIVISION.get(team)
            if division and division != current_division:
                print(f"\n##### {division} #####", file=sys.stderr)
                current_division = division
            print(f"  -- {team} --", file=sys.stderr)
            data = compile_team_season(year, team, verbose=verbose)
            write_team_year_file(team, year, data, out_dir=args.out_dir)

        if args.top_n:
            print(f"  -- leaguewide top {args.top_n}{' per position' if args.per_position else ''} --", file=sys.stderr)
            league_data = compile_league_top(year, top_n=args.top_n, per_position=args.per_position, verbose=verbose)
            write_league_year_file(year, league_data, out_dir=args.out_dir)


if __name__ == "__main__":
    main()