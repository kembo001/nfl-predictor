library(nflfastR)
library(dplyr)
library(tidyr)

# Load 2025 play-by-play data
pbp_2025 <- load_pbp(2025)

# Calculate team offensive stats
offense_stats <- pbp_2025 %>%
  filter(season_type == "REG", !is.na(posteam)) %>%
  group_by(posteam) %>%
  summarise(
    # EPA
    passing_epa = sum(epa[play_type == "pass"], na.rm = TRUE),
    rushing_epa = sum(epa[play_type == "run"], na.rm = TRUE),
    total_offensive_epa = sum(epa, na.rm = TRUE),
    
    # Pass stats
    dropbacks = sum(pass == 1, na.rm = TRUE),
    sacks_allowed = sum(sack == 1, na.rm = TRUE),
    sacks_allowed_rate = sacks_allowed / dropbacks,

    protection_rate = 1 - (sacks_allowed_rate * 2.5),
    
    .groups = "drop"
  ) %>%
  rename(team = posteam)

# Calculate team defensive stats
defense_stats <- pbp_2025 %>%
  filter(season_type == "REG", !is.na(defteam)) %>%
  group_by(defteam) %>%
  summarise(
    # Defensive EPA (from opponent's perspective, so we negate)
    defensive_epa = -sum(epa, na.rm = TRUE),
    defensive_pass_epa = -sum(epa[play_type == "pass"], na.rm = TRUE),
    defensive_rush_epa = -sum(epa[play_type == "run"], na.rm = TRUE),
    
    # Pass rush
    opp_dropbacks = sum(pass == 1, na.rm = TRUE),
    sacks = sum(sack == 1, na.rm = TRUE),
    sack_rate = sacks / opp_dropbacks,
    
    pressure_rate = sack_rate * 2.5,
    .groups = "drop"
  ) %>%
  rename(team = defteam)

# FIX: Get team records using unique game_ids
team_records <- pbp_2025 %>%
  filter(season_type == "REG", !is.na(home_team), !is.na(away_team)) %>%
  select(game_id, home_team, away_team, home_score, away_score) %>%
  distinct() %>%
  pivot_longer(
    cols = c(home_team, away_team),
    names_to = "location",
    values_to = "team"
  ) %>%
  mutate(
    team_score = ifelse(location == "home_team", home_score, away_score),
    opp_score = ifelse(location == "home_team", away_score, home_score),
    win = as.integer(team_score > opp_score),
    loss = as.integer(team_score < opp_score)
  ) %>%
  group_by(team) %>%
  summarise(
    wins = sum(win, na.rm = TRUE),
    losses = sum(loss, na.rm = TRUE),
    points_for = sum(team_score, na.rm = TRUE),
    points_against = sum(opp_score, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    win_pct = wins / (wins + losses),
    point_differential = points_for - points_against
  ) %>%
  select(team, wins, losses, win_pct, point_differential)

# Merge everything
team_stats_2025 <- team_records %>%
  left_join(offense_stats, by = "team") %>%
  left_join(defense_stats, by = "team") %>%
  mutate(
    season = 2025,
    net_epa = total_offensive_epa + defensive_epa,
  )

# Add playoff seeds for 2025-25 season
playoff_seeds <- tribble(
  ~team, ~playoff_seed, ~conference,
  # AFC
  "DEN",  1, "AFC",
  "NE", 2, "AFC",
  "JAX", 3, "AFC",
  "BAL", 4, "AFC",
  "HOU", 5, "AFC",
  "BUF", 6, "AFC",
  "LAC", 7, "AFC",
  # NFC
  "SEA", 1, "NFC",
  "CHI", 2, "NFC",
  "PHI",  3, "NFC",
  "TB",  4, "NFC",
  "LA", 5, "NFC",
  "SF", 6, "NFC",
  "GB",  7, "NFC"
)

# Final dataset
team_df_2025 <- team_stats_2025 %>%
  left_join(playoff_seeds, by = "team") %>%
  filter(!is.na(playoff_seed))

# Check the data
print(team_df_2025)

# Save to CSV
write.csv(team_df_2025, "team_stats_2025_playoffs.csv", row.names = FALSE)