import nflreadpy as nfl

# 1. Load player bio, extract gsis_id, and grab the college name
players = nfl.load_players().to_pandas()
mvs_player_row = players[players['display_name'] == "Marquez Valdes-Scantling"]

mvs_id = mvs_player_row['gsis_id'].values[0]
mvs_college = mvs_player_row['college_name'].values[0] # Extracts "South Florida"

# 2. Pull regular season offensive statistics across his career
stats = nfl.load_player_stats(seasons=list(range(2018, 2026)), summary_level="reg").to_pandas()
mvs_stats = stats[stats['player_id'] == mvs_id].copy()

# 3. Add the college column to his career stats DataFrame
mvs_stats['college_name'] = mvs_college

# 4. View key receiving metrics with college included
print(mvs_college)
