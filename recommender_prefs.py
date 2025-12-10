import pandas as pd
import numpy as np
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg2://tylerbrecker@localhost:5433/recsys_db"
)

def ask_preferences():
    print("What genres are you in the mood for today?")
    print("1) Action  2) Comedy  3) Drama  4) Horror  5) Romance  6) Sci fi  7) Animation")
    raw = input("Pick up to 3 numbers, comma separated: ")

    picks = [p.strip() for p in raw.split(",") if p.strip()]

    id_to_col = {
        "1": "is_action",
        "2": "is_comedy",
        "3": "is_drama",
        "4": "is_horror",
        "5": "is_romance",
        "6": "is_scifi",
        "7": "is_animation",
    }

    genre_weights = {col: 0.0 for col in id_to_col.values()}
    for p in picks:
        if p in id_to_col:
            genre_weights[id_to_col[p]] = 2.0   # boost chosen genres

    print("\nHow new do you want the movie?")
    print("1) Only recent (after 2000)")
    print("2) Classic and modern (after 1980)")
    print("3) No preference")
    yr_choice = input("Choose 1, 2, or 3: ").strip()

    if yr_choice == "1":
        year_min = 2000
    elif yr_choice == "2":
        year_min = 1980
    else:
        year_min = 1900

    print("\nPopularity preference?")
    print("1) Big popular hits")
    print("2) Mix of both")
    print("3) More hidden gems")
    pop_choice = input("Choose 1, 2, or 3: ").strip()

    if pop_choice == "1":
        popularity_pref = "popular"
    elif pop_choice == "3":
        popularity_pref = "hidden"
    else:
        popularity_pref = "mixed"

    prefs = {
        "genre_weights": genre_weights,
        "year_min": year_min,
        "popularity_pref": popularity_pref,
    }
    return prefs

def load_candidates():
    df = pd.read_sql("SELECT * FROM movie_features", engine)
    df = df[df["num_ratings"] >= 10].copy()
    df["avg_rating"] = df["avg_rating"].fillna(df["avg_rating"].mean())
    df["year"] = df["year"].fillna(1900).astype(int)
    return df   # includes movie_id


def score_by_preferences(df, prefs):
    scores = df["avg_rating"].astype(float).copy()

    # genre boosts
    selected_cols = []
    for col, weight in prefs["genre_weights"].items():
        if col in df.columns and weight != 0:
            selected_cols.append(col)
            scores += df[col].astype(int) * weight

    # penalize movies that do NOT match any selected genre
    if selected_cols:
        genre_match = df[selected_cols].sum(axis=1) > 0
        scores[~genre_match] -= 3.0   # tune this if needed

    # year filter and slight penalty for older than chosen
    mask_year = df["year"] >= prefs["year_min"]
    scores[~mask_year] -= 2.0

    # popularity shaping with hidden gem reinforcement
    pop = df["num_ratings"].astype(float)
    pop_pref = prefs.get("popularity_pref", "mixed")

    if pop_pref == "popular":
        # reward big hit movies
        log_pop = np.log1p(pop)
        scores += log_pop

    elif pop_pref == "hidden":
        # strong boost for movies under a rating count threshold
        hidden_threshold = 500.0  # set your own cut off

        # rarity in [0, 1]: 1 means very few ratings, 0 means at or above threshold
        rarity = (hidden_threshold - pop) / hidden_threshold
        rarity = rarity.clip(lower=0.0)

        # give a clear bonus to low count movies
        scores += rarity * 3.0

        # optional small penalty for very popular titles
        log_pop = np.log1p(pop)
        scores -= (log_pop - log_pop.min()) * 0.2

    else:
        # mixed: slight preference for popular but not huge
        log_pop = np.log1p(pop)
        scores += (log_pop - log_pop.mean()) * 0.3

    return scores



def recommend_from_preferences(prefs, top_n=50):
    df = load_candidates()
    df["pref_score"] = score_by_preferences(df, prefs)
    df = df.sort_values("pref_score", ascending=False).head(top_n)

    return df[[
        "movie_id",
        "title",
        "genres",
        "year",
        "avg_rating",
        "num_ratings",
        "pref_score",
    ]]



def main():
    prefs = ask_preferences()
    recs = recommend_from_preferences(prefs, top_n=10)
    print("\nRecommendations based on your answers:\n")
    for _, row in recs.iterrows():
        print(
            f"- {row['title']}  ({row['year']})  "
            f"rating {row['avg_rating']:.2f}  "
            f"ratings {int(row['num_ratings'])}"
        )

if __name__ == "__main__":
    main()
