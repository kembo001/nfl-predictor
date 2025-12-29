# ============================================================
# NEW: BUILD SYNTHETIC MATCHUP FEATURES (for 2025 projections)
# ============================================================

def build_matchup_row_v15(season, game_type, away_team, home_team,
                          team_df, epa_model, baselines, spread_model=None,
                           is_neutral=False,
                          include_interactions=True,
                          include_nonlinear=True,
                          include_round_features=True):
    """
    Build a single v15-style feature row for a matchup not in games_df (e.g., 2025 bracket).
    IMPORTANT: team_df should ideally include ALL 32 teams for that season so z-scores + qrank are league-based.
    """

    # Ensure z-scores exist for that season
    team_df_z = compute_season_zscores(team_df)

    away_row = team_df_z[(team_df_z["team"] == away_team) & (team_df_z["season"] == season)]
    home_row = team_df_z[(team_df_z["team"] == home_team) & (team_df_z["season"] == season)]
    if len(away_row) == 0 or len(home_row) == 0:
        return None

    away_data = away_row.iloc[0]
    home_data = home_row.iloc[0]

    away_seed = int(away_data["playoff_seed"])
    home_seed = int(home_data["playoff_seed"])

    # For neutral games, keep "home" as the better seed to make baseline_prob direction sensible
    if is_neutral and away_seed < home_seed:
        away_team, home_team = home_team, away_team
        away_data, home_data = home_data, away_data
        away_seed, home_seed = home_seed, away_seed
        if pd.notna(home_spread):
            home_spread = -float(home_spread)

    # Core stats
    away_games = float(away_data["wins"] + away_data["losses"])
    home_games = float(home_data["wins"] + home_data["losses"])
    away_pd_pg = float(away_data["point_differential"] / max(away_games, 1))
    home_pd_pg = float(home_data["point_differential"] / max(home_games, 1))
    away_net_epa = float(away_data.get("net_epa", 0) or 0)
    home_net_epa = float(home_data.get("net_epa", 0) or 0)

    matchup = compute_matchup_features(away_data, home_data)

    away_exp = float(epa_model["intercept"] + epa_model["slope"] * away_net_epa)
    home_exp = float(epa_model["intercept"] + epa_model["slope"] * home_net_epa)
    away_win_pct = float(away_data["win_pct"])
    home_win_pct = float(home_data["win_pct"])

    # Quality rank (qrank) from pd_pg within season
    season_teams = team_df_z[team_df_z["season"] == season].copy()
    season_teams["pd_pg"] = season_teams["point_differential"] / (
        (season_teams["wins"] + season_teams["losses"]).clip(lower=1)
    )
    season_teams["qrank"] = season_teams["pd_pg"].rank(ascending=False)

    away_q = season_teams[season_teams["team"] == away_team]["qrank"].values
    home_q = season_teams[season_teams["team"] == home_team]["qrank"].values
    away_quality_rank = int(away_q[0]) if len(away_q) else len(season_teams) // 2
    home_quality_rank = int(home_q[0]) if len(home_q) else len(season_teams) // 2

    away_mom = float(away_data.get("momentum_residual", 0) or 0)
    home_mom = float(home_data.get("momentum_residual", 0) or 0)

    baseline_prob = float(baseline_probability(home_seed, away_seed, is_neutral, baselines))
    baseline_logit = float(logit(np.clip(baseline_prob, 0.01, 0.99)))

    spread_offset = get_spread_offset_logit(home_spread, spread_model) if spread_model is not None else np.nan

    if pd.notna(spread_offset):
        offset_logit = float(spread_offset)
        offset_source = "spread"
        spread_prob = float(expit(spread_offset))
    else:
        offset_logit = float(baseline_logit)
        offset_source = "baseline"
        spread_prob = np.nan

    seed_diff = away_seed - home_seed

    row = {
        "season": int(season),
        "game_type": game_type,
        "away_team": away_team,
        "home_team": home_team,
        "orig_away_team": away_team,
        "orig_home_team": home_team,
        "winner": np.nan,
        "away_score": np.nan,
        "home_score": np.nan,
        "home_wins": np.nan,
        "is_neutral": 1 if is_neutral else 0,
        "away_seed": away_seed,
        "home_seed": home_seed,
        "seed_diff": seed_diff,

        "delta_net_epa": home_net_epa - away_net_epa,
        "delta_pd_pg": home_pd_pg - away_pd_pg,
        "delta_vulnerability": (away_win_pct - away_exp) - (home_win_pct - home_exp),
        "delta_underseeded": (away_seed - away_quality_rank) - (home_seed - home_quality_rank),
        "delta_momentum": home_mom - away_mom,

        "baseline_prob": baseline_prob,
        "baseline_logit": baseline_logit,
        "home_spread": home_spread,
        "spread_offset": spread_offset,
        "spread_prob": spread_prob,
        "offset_logit": offset_logit,
        "offset_source": offset_source,
    }

    row.update(matchup)

    # v15 extras
    if include_nonlinear:
        row.update(create_seed_spline_basis(seed_diff))
        row.update(create_polynomial_features(row))

    if include_interactions:
        row.update(create_interaction_features(row))

    row.update(compute_spread_features(home_spread))

    if include_round_features:
        row.update(create_round_features(game_type))

    return row


def _tossup_flag(home_prob, away_seed, home_seed, p_band=0.06):
    # Matches your “T” vibe: close probs OR tiny seed gap
    seed_gap = abs(int(away_seed) - int(home_seed))
    return (abs(float(home_prob) - 0.5) <= p_band) or (seed_gap <= 1)


def _predict_one(model_dict, gf_row, platt_params, lambda_params,
                 xgb_model=None, xgb_weight=0.30):
    # Use your v15 ensemble logic (baseline games only)
    return predict_ensemble(
        model_dict, gf_row, platt_params, lambda_params,
        xgb_model=xgb_model, xgb_weight=xgb_weight
    )


def project_playoff_bracket_v15(playoff_df, season,
                                model_dict, platt_params, lambda_params,
                                baselines, epa_model, spread_model=None,
                                xgb_model=None, xgb_weight=0.30):
    """
    playoff_df must contain at least: team, playoff_seed, conference (AFC/NFC), season.
    Ideally it contains ALL teams for that season for proper z-scores.
    """

    def conf_block(conf):
        return playoff_df[playoff_df["conference"].str.upper() == conf].copy()

    def seed_map(df_conf):
        m = {}
        for _, r in df_conf.iterrows():
            m[int(r["playoff_seed"])] = r["team"]
        return m

    def play_game(game_type, away_team, home_team, is_neutral=False, home_spread=np.nan):
        gf = build_matchup_row_v15(
            season, game_type, away_team, home_team,
            team_df=playoff_df,  # NOTE: best if this is full-season df, not just playoff teams
            epa_model=epa_model,
            baselines=baselines,
            spread_model=spread_model,
            home_spread=home_spread,
            is_neutral=is_neutral,
            include_interactions=True,
            include_nonlinear=True,
            include_round_features=True
        )
        if gf is None:
            return None

        pred = _predict_one(model_dict, gf, platt_params, lambda_params, xgb_model, xgb_weight)
        if pred is None:
            return None

        home_prob = float(pred["home_prob"])
        predicted_team = gf["home_team"] if home_prob >= 0.5 else gf["away_team"]

        tag, ud_prob, seed_gap = upset_tag_by_seed(pd.Series(gf), home_prob)

        return {
            "game_type": game_type,
            "away_team": gf["away_team"],
            "home_team": gf["home_team"],
            "away_seed": int(gf["away_seed"]),
            "home_seed": int(gf["home_seed"]),
            "home_prob": home_prob,
            "predicted": predicted_team,
            "tossup": _tossup_flag(home_prob, gf["away_seed"], gf["home_seed"]),
            "upset_tag": tag,
            "underdog_prob": float(ud_prob),
            "seed_gap": int(seed_gap),
            "src": gf["offset_source"]
        }

    def print_game(g):
        t = " T" if g["tossup"] else ""
        print(f"  {g['away_team']} ({g['away_seed']}) @ {g['home_team']} ({g['home_seed']}){t}")
        print(f"    Home Win Prob: {g['home_prob']*100:0.1f}%  |  Predicted: {g['predicted']}")
        print(f"    Seed Diff: {g['seed_gap']}  |  Src: {g['src']}")
        if g["upset_tag"]:
            print(f"    ⚠️  {g['upset_tag']} ({g['underdog_prob']*100:0.1f}%)")
        print()

    def run_conference(conf):
        dfc = conf_block(conf)
        smap = seed_map(dfc)
        # Expect seeds 1..7
        for s in range(1, 8):
            if s not in smap:
                raise ValueError(f"{conf} missing seed {s}")

        # Wild Card: 7@2, 6@3, 5@4
        wc = [
            play_game("WC", smap[7], smap[2], is_neutral=False),
            play_game("WC", smap[6], smap[3], is_neutral=False),
            play_game("WC", smap[5], smap[4], is_neutral=False),
        ]
        wc = [g for g in wc if g is not None]

        print("----------------------------------------")
        print("WILD CARD ROUND")
        print("----------------------------------------\n")
        for g in wc:
            print_game(g)

        # Winners with seeds
        winners = [(g["predicted"], g["home_seed"] if g["predicted"] == g["home_team"] else g["away_seed"]) for g in wc]

        # Seed 1 bye
        one_team = smap[1]
        one_seed = 1

        # Lowest remaining seed (largest number) plays @1
        lowest = max(winners, key=lambda x: x[1])
        other_two = [w for w in winners if w != lowest]

        div1 = play_game("DIV", lowest[0], one_team, is_neutral=False)  # away @ home (1)
        # Other matchup: better seed hosts
        (tA, sA), (tB, sB) = other_two
        if sA < sB:
            div2 = play_game("DIV", tB, tA, is_neutral=False)
        else:
            div2 = play_game("DIV", tA, tB, is_neutral=False)

        div = [g for g in [div1, div2] if g is not None]

        print("----------------------------------------")
        print("DIVISIONAL ROUND (Projected)")
        print("----------------------------------------\n")
        for g in div:
            print_game(g)

        div_winners = [(g["predicted"], g["home_seed"] if g["predicted"] == g["home_team"] else g["away_seed"]) for g in div]

        # Conference: better seed hosts
        (t1, s1), (t2, s2) = div_winners
        if s1 < s2:
            con = play_game("CON", t2, t1, is_neutral=False)
        else:
            con = play_game("CON", t1, t2, is_neutral=False)

        print("----------------------------------------")
        print("CONFERENCE CHAMPIONSHIP (Projected)")
        print("----------------------------------------\n")
        print_game(con)

        champ_team = con["predicted"]
        champ_seed = con["home_seed"] if champ_team == con["home_team"] else con["away_seed"]
        return {"champ_team": champ_team, "champ_seed": champ_seed, "games": {"wc": wc, "div": div, "con": con}}

    # Run both conferences
    afc = run_conference("AFC")
    nfc = run_conference("NFC")

    # Super Bowl: AFC @ NFC, neutral
    sb = play_game("SB", afc["champ_team"], nfc["champ_team"], is_neutral=True)
    print("----------------------------------------")
    print("SUPER BOWL (Projected)")
    print("----------------------------------------\n")
    print_game(sb)

    champ = sb["predicted"]
    print("================================================================================")
    print(f"🏆 PREDICTED SUPER BOWL CHAMPION (v15.1): {champ}")
    print("================================================================================")

    return {"AFC": afc, "NFC": nfc, "SB": sb}

