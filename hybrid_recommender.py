import numpy as np
import pandas as pd

from recommender_history import recommend_from_history
from recommender_prefs import ask_preferences, recommend_from_preferences


def min_max_normalize(series: pd.Series) -> pd.Series:
    """Normalize to 0 1 range. If constant, return zeros."""
    s = series.astype(float)
    s_min = s.min()
    s_max = s.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series(0.0, index=s.index)
    return (s - s_min) / (s_max - s_min)


def recommend_hybrid(user_id: int, alpha: float = 0.3, top_n: int = 10) -> pd.DataFrame:
    """
    Combine history based and preference based recommendations.

    alpha controls weight on preferences:
       final = alpha * pref_score_norm + (1 alpha) * history_score_norm
    """

    # 1 Get user preferences interactively
    print(f"Building hybrid recommendations for user {user_id}")
    prefs = ask_preferences()

    # 2 Get history based and preference based candidates
    hist_df = recommend_from_history(user_id, top_n=top_n * 5)
    pref_df = recommend_from_preferences(prefs, top_n=top_n * 5)

    # 3 Merge on movie_id (outer so movies from either side are allowed)
    merged = pd.merge(
        hist_df,
        pref_df,
        on="movie_id",
        how="outer",
        suffixes=("_hist", "_pref"),
    )

    # Fill missing basic fields from whichever side has them
    for col in ["title", "genres", "avg_rating", "num_ratings"]:
        col_hist = f"{col}_hist"
        col_pref = f"{col}_pref"
        if col_hist in merged.columns and col_pref in merged.columns:
            merged[col] = merged[col_hist].combine_first(merged[col_pref])
        elif col_hist in merged.columns:
            merged[col] = merged[col_hist]
        elif col_pref in merged.columns:
            merged[col] = merged[col_pref]

    # 4 Normalize scores separately
    merged["history_score"] = merged["score"].fillna(0.0)
    merged["pref_score"] = merged["pref_score"].fillna(0.0)

    merged["history_norm"] = min_max_normalize(merged["history_score"])
    merged["pref_norm"] = min_max_normalize(merged["pref_score"])

    # 5 Hybrid final score
    merged["final_score"] = (
        alpha * merged["pref_norm"] +
        (1.0 - alpha) * merged["history_norm"]
    )

    # 6 Filter and sort
    # Drop movies that have almost no signal from either side
    merged = merged[merged["final_score"] > 0]

    merged = merged.sort_values("final_score", ascending=False).head(top_n)

    return merged[[
        "movie_id",
        "title",
        "genres",
        "avg_rating",
        "num_ratings",
        "history_score",
        "pref_score",
        "final_score",
    ]]


def main():
    user_id = int(input("Enter user id: "))
    recs = recommend_hybrid(user_id, alpha=0.5, top_n=10)

    print("\nHybrid recommendations:\n")
    for _, row in recs.iterrows():
        print(
            f"- {row['title']}  "
            f"(hist {row['history_score']:.3f}, pref {row['pref_score']:.3f}, "
            f"final {row['final_score']:.3f})"
        )


if __name__ == "__main__":
    main()

