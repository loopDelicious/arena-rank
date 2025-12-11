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
