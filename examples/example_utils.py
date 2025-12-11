"""Util functions for the example notebooks"""

import pandas as pd
import matplotlib.pyplot as plt


### LMarena
def extract_metadata_features(df):
    """Extracts nested metadata dictionaries into top-level columns."""
    feature_map = {
        "sum_assistant_a_tokens": "sum_assistant_tokens_a",
        "sum_assistant_b_tokens": "sum_assistant_tokens_b",
    }
    # for dictionary columns (like header_count), sum the values
    for key in ["header_count", "bold_count", "list_count"]:
        for suffix in ["_a", "_b"]:
            col_name = f"{key}{suffix}"
            df[col_name] = df["conv_metadata"].apply(
                lambda x: sum(x[col_name].values()) if isinstance(x[col_name], dict) else 0
            )

    for old, new in feature_map.items():
        df[new] = df["conv_metadata"].apply(lambda x: x[old])
    return df


def add_style_feature_cols(df, feature_names):
    """computes normalized relative differences for the style features."""
    df = extract_metadata_features(df)

    for feature in feature_names:
        col_a, col_b = f"{feature}_a", f"{feature}_b"
        diff = df[col_a] - df[col_b]
        total = df[col_a] + df[col_b]
        total = total.replace(0, 1)
        df[feature + "_raw"] = diff / total
        df[feature] = (df[feature + "_raw"] - df[feature + "_raw"].mean()) / df[feature + "_raw"].std()

    df = df[["model_a", "model_b", "winner"] + feature_names]
    return df


def plot_leaderboard(
    results, competitors, top_n=20, item_name="Model", rating_name="Arena Score", title="Style-Control Leaderboard"
):
    ""
    leaderboard_df = pd.DataFrame(
        {
            item_name: competitors,
            "Rating": results["ratings"],
            "Lower": results["rating_lower"],
            "Upper": results["rating_upper"],
        }
    )

    # sort by rating
    leaderboard_df = leaderboard_df.sort_values(by="Rating", ascending=False).reset_index(drop=True)

    # calculate error bar sizes
    leaderboard_df["error_lower"] = leaderboard_df["Rating"] - leaderboard_df["Lower"]
    leaderboard_df["error_upper"] = leaderboard_df["Upper"] - leaderboard_df["Rating"]
    plot_df = leaderboard_df.head(top_n)
    plt.figure(figsize=(14, 8))
    plt.errorbar(
        x=plot_df["Model"],
        y=plot_df["Rating"],
        yerr=[plot_df["error_lower"], plot_df["error_upper"]],
        fmt="o",
        color="royalblue",
        ecolor="gray",
        capsize=3,
        markersize=6,
    )
    plt.title(f"{title} (Top {top_n})", fontsize=16)
    plt.ylabel(rating_name, fontsize=12)
    plt.xlabel(f"{item_name} Name", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.tight_layout()
    plt.show()


### Melee


def plot_melee_bump_chart(df_all_years, message_type="custom"):
    PLAYER_COLOR_MAP = {
        "Mang0": "#D55E00",
        "Armada": "#0072B2",
        "Hungrybox": "#E69F00",
        "Mew2King": "#009E73",
        "Ken": "#CC79A7",
        "Zain": "#F0E442",
        "PPMD": "#56B4E9",
        "Cody Schwab": "#000000",
        "Leffen": "#0000FF",
        "Azen": "#00FF00",
        "ChuDat": "#FF00E4",
        "PC Chris": "#1A0078",
        "Moky": "#BFC98A",
        "Plup": "#FF005D",
        "AMSa": "#505000",
        "KoreanDJ": "#5D0000",
        "Jmook": "#AE78FF",
        "Isai": "#93FF86",
        "Cort": "#500D43",
        "Axe": "#00FFFF",
        "DaShizWiz": "#FFA186",
        "Sastopher": "#6B6B6B",
        "Wizzrobe": "#358600",
    }

    df_all_years = df_all_years.copy()
    df_all_years["Rank"] = df_all_years.groupby("Year")["Rating"].rank(ascending=False, method="first")
    df_top = df_all_years[df_all_years["Rank"] <= 5].copy()

    # Calculate sort order only (colors are now static)
    df_top["Score"] = 1 / df_top["Rank"]
    sorted_players = df_top.groupby("Competitor")["Score"].sum().sort_values(ascending=False).index.tolist()

    plt.style.use("default")
    plt.figure(figsize=(24, 8))
    unique_years = sorted(df_top["Year"].unique())

    plt.axvspan(2020 - 0.5, 2021 + 0.5, color="#d4d4d4", alpha=0.6, zorder=1)
    text = (
        "Due to Covid the data is mostly from online tournaments and is of lower quality"
        if message_type == "custom"
        else "Due to Covid no official ranks were released"
    )
    plt.text(
        (2020 + 2021) / 2,
        5.8,
        text,
        fontsize=9,
        color="black",
        ha="center",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9, ec="grey", linewidth=0.5),
        zorder=10,
    )

    for player in sorted_players:
        player_data = df_top[df_top["Competitor"] == player].sort_values("Year")
        if player_data.empty:
            continue

        years, ranks = player_data["Year"].values, player_data["Rank"].values
        segments = []
        if len(years) > 0:
            cx, cy = [years[0]], [ranks[0]]
            for i in range(1, len(years)):
                if years[i] == years[i - 1] + 1:
                    cx.append(years[i])
                    cy.append(ranks[i])
                else:
                    segments.append((cx, cy))
                    cx, cy = [years[i]], [ranks[i]]
            segments.append((cx, cy))

        # Use static color map, default to black if player not in map
        color = PLAYER_COLOR_MAP.get(player, "#000000")

        for x_seg, y_seg in segments:
            plt.plot(x_seg, y_seg, marker="o", linewidth=5, color=color, zorder=5)
            last_x, last_y = x_seg[-1], y_seg[-1]
            offset = 0.1
            if len(y_seg) >= 2:
                final_y = last_y - offset if last_y <= y_seg[-2] else last_y + offset
            else:
                final_y = last_y - offset
            plt.text(
                last_x, final_y, player, color=color, fontweight="bold", fontsize=11, ha="center", va="center", zorder=6
            )

    plt.gca().invert_yaxis()
    plt.yticks(range(1, 6))
    xtick_labels = [str(y) + ("*" if y in [2020, 2021] else "") for y in unique_years]
    plt.xticks(unique_years, xtick_labels, fontsize=12)
    plt.ylabel("Rank (1=Highest)")
    plt.xlabel("Year")
    extra = "(SSBMRank/RetroSSBMRank)" if message_type == "official" else "(Bradley-Terry Rankings)"
    plt.title(f"Top 5 Melee Players Over Time {extra}", fontsize=14, fontweight="bold")
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    if unique_years:
        plt.xlim(min(unique_years) - 0.5, max(unique_years) + 0.5)
    plt.ylim(5.5, 0.7)
    plt.tight_layout()
    plt.show()


def filter_melee_matches(df):
    # reduces noise by applying two filters:
    # 1. Remove players with few unique opponents, this helps prevent issues due to disconnected components
    # 2. Remove players who have only wins or only losses, these cases would cause BT ratings to to to infinity or -infinity
    opponents_1 = df.groupby("competitor_1")["competitor_2"].nunique()
    opponents_2 = df.groupby("competitor_2")["competitor_1"].nunique()
    total_opponents = opponents_1.add(opponents_2, fill_value=0)
    total_opponent_thresh = total_opponents.quantile(0.70)
    total_opponent_thresh = max(total_opponent_thresh, 3)
    players_with_min_opponents = total_opponents[total_opponents >= total_opponent_thresh].index
    df = df[df["competitor_1"].isin(players_with_min_opponents) & df["competitor_2"].isin(players_with_min_opponents)]

    # iteratively remove players with only wins or only losses
    # apply several times since removing one player can cause another to have only wins or losses
    for _ in range(5):
        player_wins = (
            df.groupby("competitor_1")["outcome"]
            .sum()
            .add(df.groupby("competitor_2")["outcome"].apply(lambda x: (x == 0).sum()), fill_value=0)
        )
        player_losses = (
            df.groupby("competitor_1")["outcome"]
            .apply(lambda x: (x == 0).sum())
            .add(df.groupby("competitor_2")["outcome"].sum(), fill_value=0)
        )
        players_with_both = player_wins[(player_wins >= 1) & (player_losses >= 1)].index
        df = df[(df["competitor_1"].isin(players_with_both)) & (df["competitor_2"].isin(players_with_both))]
    return df


def filter_melee_leaderboard(leaderboard_df, match_df):
    # even after filtering matches, some players may still have extreme records, or very few matches and should be removed from the leaderboard
    # filter based on number of unique opponents in the match_df and on total matches played
    # heuristics are applied based on how many total matches are in the match_df
    opponents_1 = match_df.groupby("competitor_1")["competitor_2"].nunique()
    opponents_2 = match_df.groupby("competitor_2")["competitor_1"].nunique()
    total_opponents = opponents_1.add(opponents_2, fill_value=0)
    players_with_min_opponents = set(total_opponents[total_opponents >= 10].index)

    counts = leaderboard_df.head(100).set_index("Competitor")["Matches Played"]
    # heuristics for minimum matches played based on total matches that year
    total_matches = len(match_df)
    if total_matches < 1000:
        lower_bound = 10
    elif total_matches < 4000:
        lower_bound = 20
    else:
        lower_bound = 30
    # only keep players who have played at least the 60th percentile of matches played among the top 100 players
    threshold = min(max(counts.quantile(0.6), lower_bound), 35)
    players_with_min_matches = set(leaderboard_df[leaderboard_df["Matches Played"] >= threshold]["Competitor"].tolist())
    leaderboard_players = players_with_min_opponents.intersection(players_with_min_matches)
    leaderboard_players.discard("Zion")  # Zion is a banned player and ineligible for ranking
    leaderboard_df = leaderboard_df[leaderboard_df["Competitor"].isin(leaderboard_players)].reset_index(drop=True)
    leaderboard_df["Competitor"] = leaderboard_df["Competitor"].apply(lambda x: x.replace(" (Melee player)", ""))
    return leaderboard_df


def convert_ssbmrank_dict_to_plotting_df(ranking_dict):
    data = []
    for year, players in ranking_dict.items():
        if players == ["No Official Ranking"]:
            data.append({"Competitor": f"Placeholder-{year}", "Year": year, "Rating": 0})
            continue
        for i, player in enumerate(players):
            if not player:
                print(f"Empty player name for year {year}, rank {i + 1}")
            rank = i + 1
            rating = 6 - rank
            if rank <= 5:
                data.append({"Competitor": player, "Year": year, "Rating": rating})
    df = pd.DataFrame(data)
    min_year = min(ranking_dict.keys())
    max_year = max(ranking_dict.keys())
    all_years_range = range(min_year, max_year + 1)
    all_years_df = pd.DataFrame({"Year": list(all_years_range)})
    df_all_years = all_years_df.merge(df, on="Year", how="left")
    return df_all_years
