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

--colleges compiles career files per college, e.g. College_TEXAS_Stats.json
(see compile_college_positions() below).

--draft-rounds compiles files per NFL draft round, e.g. DraftRound1.json,
covering every skill player ever drafted in that round -- see
compile_draft_round() below. This uses nflreadpy.load_draft_picks(), which
carries each player's *career* totals (as recorded by Pro Football
Reference) directly on the draft-pick row, so no per-season join against
load_player_stats() is needed for this feature.

--conferences compiles career files per college conference, e.g.
Conference_SEC_Stats.json, covering every player from every school in that
conference -- see compile_conference_positions() below. nflreadpy's
load_players() table has a college_conference column, but it's null for a
lot of individual player rows (especially older ones), so this falls back
to each school's most commonly recorded conference when a specific player's
own row doesn't have one. See the fallback's docstring for the realignment
caveat that comes with that (Texas/Oklahoma -> SEC, USC/UCLA -> Big Ten,
etc.).

Install:
    pip install nflreadpy --break-system-packages

Usage (mirrors the original script):
    python nfl_roster_stats.py 2025 --divisions "NFC North"
    python nfl_roster_stats.py 2025 --teams GB,KC,DAL
    python nfl_roster_stats.py all --start 2025 --end 2016 --divisions all
    python nfl_roster_stats.py --colleges "Texas,Ohio State"
    python nfl_roster_stats.py --draft-rounds "1,2,3"
    python nfl_roster_stats.py --draft-rounds all --draft-start 1990
    python nfl_roster_stats.py --conferences "SEC,Big Ten,ACC,Big 12"
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
# College conferences. nflverse spells these out in full ("Southeastern
# Conference", "Big Twelve Conference", ...) so we map common shorthand the
# user would actually type on the command line to those full names.
# ---------------------------------------------------------------------------
CONFERENCE_ALIASES = {
    "SEC": "Southeastern Conference",
    "BIG TEN": "Big Ten Conference",
    "BIGTEN": "Big Ten Conference",
    "B1G": "Big Ten Conference",
    "BIG 12": "Big Twelve Conference",
    "BIG12": "Big Twelve Conference",
    "BIG TWELVE": "Big Twelve Conference",
    "ACC": "Atlantic Coast Conference",
    "PAC 12": "Pacific Twelve Conference",
    "PAC12": "Pacific Twelve Conference",
    "PAC 10": "Pacific Ten Conference",
    "PAC10": "Pacific Ten Conference",
    "BIG EAST": "Big East",
    "BIGEAST": "Big East",
    "CUSA": "Conference USA",
    "C USA": "Conference USA",
    "CONFERENCE USA": "Conference USA",
    "MOUNTAIN WEST": "Mountain West Conference",
    "MWC": "Mountain West Conference",
    "AAC": "American Athletic Conference",
    "AMERICAN": "American Athletic Conference",
    "MAC": "Mid-American Conference",
    "WAC": "Western Athletic Conference",
    "SUN BELT": "Sun Belt Conference",
    "SUNBELT": "Sun Belt Conference",
    "IVY": "Ivy League",
    "IVY LEAGUE": "Ivy League",
    "MEAC": "Mid-Eastern Athletic Conference",
    "SWAC": "Southwestern Athletic Conference",
    "CAA": "Colonial Athletic Association",
    "PATRIOT": "Patriot League",
    "PATRIOT LEAGUE": "Patriot League",
    "BIG SKY": "Big Sky Conference",
    "SOUTHERN": "Southern Conference",
    "SOUTHLAND": "Southland Conference",
    "OVC": "Ohio Valley Conference",
    "UAC": "United Athletic Conference",
}


def normalize_conference_token(s):
    return re.sub(r"[\s_-]+", " ", (s or "").strip()).upper()


def resolve_conference_name(token, known_conferences):
    """Match user input (an alias or a partial/full name) to a canonical
    conference name that actually appears in nflverse's data."""
    norm = normalize_conference_token(token)

    aliased = CONFERENCE_ALIASES.get(norm)
    if aliased and aliased in known_conferences:
        return aliased

    for candidate in known_conferences:
        if normalize_conference_token(candidate) == norm:
            return candidate

    substring_matches = [c for c in known_conferences if norm in normalize_conference_token(c)]
    if len(substring_matches) == 1:
        return substring_matches[0]
    if len(substring_matches) > 1:
        raise ValueError(f"'{token}' matches multiple conferences: {substring_matches}. Be more specific.")

    raise ValueError(f"Unknown conference '{token}'. Known conferences: {sorted(known_conferences)}")


def conference_file_slug(s):
    """'SEC' -> 'SEC', 'Big Ten' -> 'BIGTEN' -- uses whatever the user typed, not the resolved full name."""
    return re.sub(r"[^A-Za-z0-9]", "", s or "").upper()


# ---------------------------------------------------------------------------
# Position grouping + stat key mapping (same shape as the original script)
# ---------------------------------------------------------------------------
POSITION_GROUP_MAP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
GROUP_POSITIONS = {"QB": ["QB"], "RB": ["RB", "FB"], "WR": ["WR"], "TE": ["TE"]}

WANTED_KEYS_BY_GROUP = {
    "QB": {"yds", "td", "int"},
    "RB": {"ryds", "rtd", "recyds", "rectd", "rec"},
    "WR": {"recyds", "rectd", "rec"},
    "TE": {"recyds", "rectd", "rec"},
}

# nflverse column name -> our short stat key (season-level, from load_player_stats())
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

# load_draft_picks() column name -> our short stat key (career totals, from PFR)
DRAFT_STAT_COL_MAP = {
    "pass_yards": "yds",
    "pass_tds": "td",
    "pass_ints": "int",
    "rush_yards": "ryds",
    "rush_tds": "rtd",
    "rec_yards": "recyds",
    "rec_tds": "rectd",
    "receptions": "rec",
}


# ---------------------------------------------------------------------------
# Data loading (one call per season/table, reused everywhere it's needed)
# ---------------------------------------------------------------------------
_roster_cache = {}
_stats_cache = {}
_draft_picks_cache = None
_players_table_cache = None
_players_cache = None
_conference_lookups_cache = None


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
        # NOTE: nflverse's play-by-play-derived stats only go back to 1999 --
        # seasons before that will fail to download (404) even though
        # load_rosters() covers much older years.
        df = nfl.load_player_stats(seasons=[year], summary_level="reg")
        _stats_cache[year] = df
    return _stats_cache[year]


def load_draft_picks_table(verbose=True):
    global _draft_picks_cache
    if _draft_picks_cache is None:
        if verbose:
            print("  loading draft picks via nflreadpy.load_draft_picks()...", file=sys.stderr)
        _draft_picks_cache = nfl.load_draft_picks()
    return _draft_picks_cache


def load_players_table(verbose=True):
    """The single static nflreadpy.load_players() biographical table (one row
    per player), cached once and reused by both the college and conference
    lookups below."""
    global _players_table_cache
    if _players_table_cache is None:
        if verbose:
            print("  loading player bios via nflreadpy.load_players()...", file=sys.stderr)
        _players_table_cache = nfl.load_players()
    return _players_table_cache


def load_college_lookup(verbose=True):
    """gsis_id -> college_name (this is a one-row-per-player biographical
    table, unlike load_rosters() which is season-by-season and has spotty
    college data for older seasons)."""
    global _players_cache
    if _players_cache is None:
        players = load_players_table(verbose=verbose)
        _players_cache = dict(zip(players["gsis_id"].to_list(), players["college_name"].to_list()))
    return _players_cache


def load_conference_lookups(verbose=True):
    """
    Returns (per_player_conference, college_mode_conference):

      - per_player_conference: gsis_id -> college_conference exactly as
        recorded for that specific player. This field is fairly sparse,
        especially for older players -- plenty of rows are null even for
        well-known SEC/Big Ten schools.
      - college_mode_conference: college_name -> whichever conference is
        most commonly recorded for that school across every player who does
        have a non-null value. Used as a fallback when a player's own
        conference is null.

    The fallback is an approximation: for schools that changed conferences
    during realignment (Texas/Oklahoma -> SEC, USC/UCLA -> Big Ten, etc.),
    it reflects whichever conference is best-represented across that
    school's history in this table -- not necessarily the conference that
    specific player's teams were actually in.
    """
    global _conference_lookups_cache
    if _conference_lookups_cache is None:
        players = load_players_table(verbose=verbose)
        per_player = dict(zip(players["gsis_id"].to_list(), players["college_conference"].to_list()))

        non_null = players.filter(pl.col("college_conference").is_not_null())
        counts = (
            non_null.group_by(["college_name", "college_conference"])
            .len()
            .sort("len", descending=True)
        )
        college_mode = {}
        for row in counts.iter_rows(named=True):
            college = row["college_name"]
            if college not in college_mode:
                college_mode[college] = row["college_conference"]

        _conference_lookups_cache = (per_player, college_mode)
    return _conference_lookups_cache


def known_conferences(verbose=True):
    per_player, college_mode = load_conference_lookups(verbose=verbose)
    values = set(v for v in per_player.values() if v)
    values.update(college_mode.values())
    return values


def resolve_player_conference(player_id, college_lookup, per_player_conf, college_mode_conf):
    conf = per_player_conf.get(player_id)
    if conf:
        return conf
    college = college_lookup.get(player_id)
    if college:
        return college_mode_conf.get(college)
    return None


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


def _row_to_entry(row, group, team_col, yy, col_map=STAT_COL_MAP):
    wanted = WANTED_KEYS_BY_GROUP[group]
    player_stats = {}
    for col, short_key in col_map.items():
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


def normalize_college_name(s):
    """'Ohio State' -> 'ohio state', 'Texas A&amp;M' -> 'texas a&m' -- for matching."""
    if s is None:
        return None
    s = s.replace("&amp;", "&")
    return re.sub(r"\s+", " ", s.strip()).lower()


def college_file_slug(s):
    """'Ohio State' -> 'OHIOSTATE', 'Miami' -> 'MIAMI', 'Texas A&M' -> 'TEXASAM'"""
    s = s.replace("&amp;", "&")
    return re.sub(r"[^A-Za-z0-9]", "", s).upper()


def compile_college_positions(college_name, start_year, end_year, top_n=None, verbose=True):
    """
    Best players from a given college across every season in [start_year, end_year],
    grouped by position, pulled from whatever NFL team(s) they played for that year.
    Unlike the per-team files, the same player can appear multiple times here -- once
    per season they posted stats -- since the point is to show their career across
    different teams/years, not a single-season snapshot.
    """
    result = {"QB": [], "RB": [], "WR": [], "TE": []}
    college_lookup = load_college_lookup(verbose=verbose)
    target = normalize_college_name(college_name)

    lo, hi = sorted((start_year, end_year))
    for year in range(hi, lo - 1, -1):
        stats = load_season_stats(year, verbose=verbose)
        id_col = "player_id" if "player_id" in stats.columns else "gsis_id"
        team_col = "recent_team" if "recent_team" in stats.columns else "team"
        yy = str(year)[-2:]

        for row in stats.iter_rows(named=True):
            player_id = row.get(id_col)
            if normalize_college_name(college_lookup.get(player_id)) != target:
                continue

            group = POSITION_GROUP_MAP.get(row.get("position"))
            if not group:
                continue

            entry = _row_to_entry(row, group, team_col, yy)
            if all(v == 0 for v in entry["stats"].values()):
                continue

            if verbose:
                print(f"    + [{group}] {entry['name']} ({entry['team']}): {entry['stats']}", file=sys.stderr)
            result[group].append(entry)

    if top_n:
        for group in result:
            result[group].sort(key=lambda e: sum(e["stats"].values()), reverse=True)
            result[group] = result[group][:top_n]

    return result


def compile_conference_positions(resolved_conference, start_year, end_year, top_n=None, verbose=True):
    """
    Same idea as compile_college_positions(), but for every school in a
    given conference instead of a single college. resolved_conference must
    already be a canonical name from known_conferences() -- resolve it with
    resolve_conference_name() before calling this.
    """
    result = {"QB": [], "RB": [], "WR": [], "TE": []}
    college_lookup = load_college_lookup(verbose=verbose)
    per_player_conf, college_mode_conf = load_conference_lookups(verbose=verbose)

    lo, hi = sorted((start_year, end_year))
    for year in range(hi, lo - 1, -1):
        stats = load_season_stats(year, verbose=verbose)
        id_col = "player_id" if "player_id" in stats.columns else "gsis_id"
        team_col = "recent_team" if "recent_team" in stats.columns else "team"
        yy = str(year)[-2:]

        for row in stats.iter_rows(named=True):
            player_id = row.get(id_col)
            conf = resolve_player_conference(player_id, college_lookup, per_player_conf, college_mode_conf)
            if conf != resolved_conference:
                continue

            group = POSITION_GROUP_MAP.get(row.get("position"))
            if not group:
                continue

            entry = _row_to_entry(row, group, team_col, yy)
            if all(v == 0 for v in entry["stats"].values()):
                continue

            if verbose:
                print(f"    + [{group}] {entry['name']} ({entry['team']}): {entry['stats']}", file=sys.stderr)
            result[group].append(entry)

    if top_n:
        for group in result:
            result[group].sort(key=lambda e: sum(e["stats"].values()), reverse=True)
            result[group] = result[group][:top_n]

    return result


def parse_draft_rounds(rounds_arg, verbose=True):
    """
    Parse --draft-rounds into a list of round numbers. 'all' expands to
    every round number that actually appears in load_draft_picks() (the
    draft had more than 7 rounds before 1994, so this can go well past 7
    if you don't restrict it with --draft-start).
    """
    tokens = [t.strip() for t in rounds_arg.split(",") if t.strip()]
    if any(t.lower() == "all" for t in tokens):
        picks = load_draft_picks_table(verbose=verbose)
        return sorted(set(picks["round"].to_list()))

    rounds = [int(t) for t in tokens]
    seen = set()
    deduped = []
    for r in rounds:
        if r not in seen:
            seen.add(r)
            deduped.append(r)
    return deduped


def compile_draft_round(round_num, start_year=None, end_year=None, top_n=None, verbose=True):
    """
    Every skill player ever drafted in `round_num`, grouped by position,
    using their *career* totals as recorded by Pro Football Reference (via
    nflreadpy.load_draft_picks() -- no per-season stats join needed, since
    the draft-pick row already carries the player's full career line).

    start_year/end_year optionally restrict which draft *classes* count
    (e.g. start_year=1990 to only include players drafted in 1990 or
    later). Both are inclusive; leave either as None to leave that end of
    the range open.

    Players with no recorded career production in any tracked stat (busts
    who never played, or very recent rookies with no stats logged yet) are
    skipped, same as the roster-file behavior elsewhere in this script.
    """
    result = {"QB": [], "RB": [], "WR": [], "TE": []}
    picks = load_draft_picks_table(verbose=verbose)

    pool = picks.filter(pl.col("round") == round_num)
    if start_year is not None:
        pool = pool.filter(pl.col("season") >= start_year)
    if end_year is not None:
        pool = pool.filter(pl.col("season") <= end_year)

    for row in pool.iter_rows(named=True):
        group = POSITION_GROUP_MAP.get(row.get("position"))
        if not group:
            continue  # not a skill position we track (OL, DL, DB, K, etc.)

        wanted = WANTED_KEYS_BY_GROUP[group]
        player_stats = {}
        for col, short_key in DRAFT_STAT_COL_MAP.items():
            if short_key not in wanted:
                continue
            value = row.get(col)
            if value is None:
                value = 0
            elif isinstance(value, float) and value.is_integer():
                value = int(value)
            player_stats[short_key] = value

        if all(v == 0 for v in player_stats.values()):
            continue  # no career production -- e.g. a bust who never played, or a rookie with no stats yet

        yy = str(row.get("season"))[-2:]
        team = row.get("team") or "FA"
        name = row.get("pfr_player_name") or "Unknown Player"
        entry = {"name": name, "team": f"{team} '{yy}", "stats": player_stats}

        if verbose:
            print(f"    + [{group}] Rd{round_num} Pick {row.get('pick')} "
                  f"({row.get('season')}): {name} ({entry['team']}): {player_stats}", file=sys.stderr)

        result[group].append(entry)

    if top_n:
        for group in result:
            result[group].sort(key=lambda e: sum(e["stats"].values()), reverse=True)
            result[group] = result[group][:top_n]

    return result


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


def write_college_file(college_name, data, out_dir="data"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    fname = f"College_{college_file_slug(college_name)}_Stats.json"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    counts = {k: len(v) for k, v in data.items()}
    print(f"  -> wrote {out_path} ({counts})", file=sys.stderr)
    return out_path


def write_conference_file(conference_token, data, out_dir="data"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    fname = f"Conference_{conference_file_slug(conference_token)}_Stats.json"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    counts = {k: len(v) for k, v in data.items()}
    print(f"  -> wrote {out_path} ({counts})", file=sys.stderr)
    return out_path


def write_draft_round_file(round_num, data, out_dir="data"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    fname = f"DraftRound{round_num}.json"
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
    parser.add_argument("year", type=str, nargs="?", default=None,
                         help="Season year (e.g. 2025), or 'all' to compile every season in --start..--end. "
                              "Not required when using --colleges or --draft-rounds by themselves.")
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
    parser.add_argument("--colleges", type=str, default=None,
                         help="Comma-separated college names (as nflverse spells them, e.g. "
                              "'Texas,Miami,Ohio State') to compile career files for, one file per "
                              "college covering --college-start..--college-end, e.g. College_TEXAS_Stats.json. "
                              "Independent of --teams/--divisions/--top-n/--draft-rounds; doesn't require "
                              "the positional 'year' argument.")
    parser.add_argument("--college-start", type=int, default=2025, help="Newest year for --colleges (default: 2025)")
    parser.add_argument("--college-end", type=int, default=1999, help="Oldest year for --colleges (default: 1999)")
    parser.add_argument("--college-top-n", type=int, default=None, metavar="N",
                         help="Cap each position to the top N player-seasons per college (ranked by total "
                              "of the tracked stat fields). Default: no cap, include every non-zero season.")
    parser.add_argument("--draft-rounds", type=str, default=None,
                         help="Comma-separated draft round numbers to compile into 'DraftRound{N}.json' "
                              "files, e.g. '1,2,3'. Use 'all' for every round found in the data "
                              "(historically up to 12 in very old drafts, 7 since 1994 -- combine with "
                              "--draft-start/--draft-end to limit that). Pulls each player's *career* "
                              "totals directly from nflreadpy.load_draft_picks() -- independent of "
                              "--teams/--divisions/--colleges and doesn't require the positional 'year' "
                              "argument.")
    parser.add_argument("--draft-start", type=int, default=None,
                         help="Oldest draft class season to include for --draft-rounds (default: earliest available, 1980)")
    parser.add_argument("--draft-end", type=int, default=None,
                         help="Newest draft class season to include for --draft-rounds (default: latest available)")
    parser.add_argument("--draft-top-n", type=int, default=None, metavar="N",
                         help="Cap each position group within a draft round to the top N players (ranked by "
                              "total of the tracked career stat fields). Default: include everyone with any "
                              "non-zero career production.")
    parser.add_argument("--conferences", type=str, default=None,
                         help="Comma-separated college conferences to compile career files for, one file per "
                              "conference covering --conference-start..--conference-end, e.g. "
                              "'SEC,Big Ten,ACC,Big 12' -> Conference_SEC_Stats.json, Conference_BIGTEN_Stats.json, "
                              "etc. Accepts common abbreviations (SEC, ACC, Big 12, Pac-12, AAC, MAC, WAC, Sun "
                              "Belt, Mountain West, CUSA, Ivy, SWAC, MEAC, CAA, Patriot, Big Sky...) or the full "
                              "nflverse name. Independent of --teams/--divisions/--colleges/--draft-rounds; "
                              "doesn't require the positional 'year' argument.")
    parser.add_argument("--conference-start", type=int, default=2025, help="Newest year for --conferences (default: 2025)")
    parser.add_argument("--conference-end", type=int, default=1999, help="Oldest year for --conferences (default: 1999)")
    parser.add_argument("--conference-top-n", type=int, default=None, metavar="N",
                         help="Cap each position to the top N player-seasons per conference (ranked by total "
                              "of the tracked stat fields). Default: no cap, include every non-zero season.")
    args = parser.parse_args()

    if args.top_n_only and args.top_n is None:
        parser.error("--top-n-only requires --top-n.")

    if args.year is None and not args.colleges and not args.draft_rounds and not args.conferences:
        parser.error("Must specify a year (or 'all'), unless using --colleges, --draft-rounds, or --conferences.")

    verbose = not args.quiet

    if args.colleges:
        colleges = [c.strip() for c in args.colleges.split(",") if c.strip()]
        print(f"Compiling college files for: {colleges} ({args.college_start}..{args.college_end})", file=sys.stderr)
        for college in colleges:
            print(f"\n=== {college} ===", file=sys.stderr)
            data = compile_college_positions(
                college, args.college_start, args.college_end, top_n=args.college_top_n, verbose=verbose
            )
            write_college_file(college, data, out_dir=args.out_dir)

    if args.conferences:
        tokens = [c.strip() for c in args.conferences.split(",") if c.strip()]
        available = known_conferences(verbose=verbose)
        print(f"Compiling conference files for: {tokens} ({args.conference_start}..{args.conference_end})", file=sys.stderr)
        for token in tokens:
            try:
                resolved = resolve_conference_name(token, available)
            except ValueError as err:
                parser.error(str(err))
            print(f"\n=== {token} -> {resolved} ===", file=sys.stderr)
            data = compile_conference_positions(
                resolved, args.conference_start, args.conference_end, top_n=args.conference_top_n, verbose=verbose
            )
            write_conference_file(token, data, out_dir=args.out_dir)

    if args.draft_rounds:
        try:
            rounds = parse_draft_rounds(args.draft_rounds, verbose=verbose)
        except ValueError:
            parser.error(f"--draft-rounds must be comma-separated integers or 'all', got '{args.draft_rounds}'")
        print(f"Compiling draft rounds {rounds} "
              f"(seasons {args.draft_start or 'earliest'}..{args.draft_end or 'latest'})...", file=sys.stderr)
        for rnd in rounds:
            print(f"\n=== Draft Round {rnd} ===", file=sys.stderr)
            data = compile_draft_round(
                rnd, start_year=args.draft_start, end_year=args.draft_end,
                top_n=args.draft_top_n, verbose=verbose,
            )
            write_draft_round_file(rnd, data, out_dir=args.out_dir)

    if args.year is None:
        return

    teams = []
    if not args.top_n_only:
        try:
            teams = collect_teams(args.teams, args.divisions)
        except ValueError as err:
            parser.error(str(err))

        if not teams:
            parser.error("Must specify --teams and/or --divisions (or --divisions all), unless using --top-n-only.")

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