import os

from flask import Flask, render_template, request, redirect, url_for, jsonify
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import requests
from sqlalchemy import create_engine, text

from recommender_history import recommend_from_history
from recommender_prefs import score_by_preferences

# -------------------------------------------------------------------
# Env and DB setup
# -------------------------------------------------------------------

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
engine = create_engine(
    "postgresql+psycopg2://tylerbrecker@localhost:5433/recsys_db"
)

app = Flask(__name__)

# Local "owner" of this app
MY_USER_ID = int(os.getenv("LOCAL_USER_ID", "9999999"))


def ensure_local_user():
    """Make sure MY_USER_ID exists in the users table."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO users (user_id)
                VALUES (:uid)
                ON CONFLICT (user_id) DO NOTHING
            """),
            {"uid": MY_USER_ID},
        )


ensure_local_user()

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def min_max_normalize(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    s_min = s.min()
    s_max = s.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return pd.Series(0.0, index=s.index)
    return (s - s_min) / (s_max - s_min)


def get_poster_url(tmdb_id):
    """Return poster URL from TMDB or None."""
    if not TMDB_API_KEY or pd.isna(tmdb_id):
        return None
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
            params={"api_key": TMDB_API_KEY},
            timeout=3,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return None
        return f"https://image.tmdb.org/t/p/w342{poster_path}"
    except Exception:
        return None


# cache for TMDB language lookups
LANG_CACHE = {}


def get_original_language(tmdb_id):
    """Return TMDB original_language (like 'en', 'fr') or None."""
    if not TMDB_API_KEY or pd.isna(tmdb_id):
        return None
    try:
        tmdb_id_int = int(tmdb_id)
    except (TypeError, ValueError):
        return None

    if tmdb_id_int in LANG_CACHE:
        return LANG_CACHE[tmdb_id_int]

    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id_int}",
            params={"api_key": TMDB_API_KEY},
            timeout=3,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        lang = data.get("original_language")
        LANG_CACHE[tmdb_id_int] = lang
        return lang
    except Exception:
        return None


def load_candidates():
    df = pd.read_sql("SELECT * FROM movie_features_with_links", engine)
    df = df[df["num_ratings"] >= 10].copy()
    df["avg_rating"] = df["avg_rating"].fillna(df["avg_rating"].mean())
    df["year"] = df["year"].fillna(1900).astype(int)
    return df


def get_user_rated_movie_ids(user_id: int):
    """Return a set of movie_ids that this user has already rated."""
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT movie_id FROM ratings WHERE user_id = :uid"),
            {"uid": user_id},
        ).fetchall()
    return {r[0] for r in rows}


# -------------------------------------------------------------------
# Core hybrid recommender
# -------------------------------------------------------------------

def compute_hybrid_recs(
    user_id,
    genres_chosen,
    year_min,
    popularity_pref,
    alpha,
    foreign_choice,
    top_n=10,
):
    # If no TMDB key, foreign filter cannot work. Treat as "any".
    effective_foreign_choice = foreign_choice
    if not TMDB_API_KEY:
        effective_foreign_choice = "any"

    # 1) base candidate pool
    candidates = load_candidates()

    # 1a) drop movies this user already rated
    rated_ids = get_user_rated_movie_ids(user_id)
    if rated_ids:
        candidates = candidates[~candidates["movie_id"].isin(rated_ids)].copy()

    # 2) history based scores, but do NOT decide candidates here
    hist_df = recommend_from_history(user_id, top_n=500)
    # just keep movie_id and score
    hist_df = hist_df[["movie_id", "score"]].drop_duplicates("movie_id")

    # 3) preference based scoring on the candidate set
    genre_map = {
        "Action": "is_action",
        "Comedy": "is_comedy",
        "Drama": "is_drama",
        "Horror": "is_horror",
        "Romance": "is_romance",
        "Sci fi": "is_scifi",
        "Animation": "is_animation",
    }
    genre_weights = {col: 0.0 for col in genre_map.values()}
    for g in genres_chosen:
        col = genre_map[g]
        genre_weights[col] = 2.0

    prefs = {
        "genre_weights": genre_weights,
        "year_min": year_min,
        "popularity_pref": popularity_pref,
    }

    # optional hard filter on genres if any chosen
    if genres_chosen:
        cols_used = [genre_map[g] for g in genres_chosen]
        mask = candidates[cols_used].sum(axis=1) > 0
        candidates = candidates[mask].copy()

    # apply preference scoring
    candidates["pref_score"] = score_by_preferences(candidates, prefs)

    # 4) join history scores onto the candidates
    merged = candidates.merge(hist_df, on="movie_id", how="left")
    merged["history_score"] = merged["score"].fillna(0.0)
    if "score" in merged.columns:
        merged.drop(columns=["score"], inplace=True)

    # 5) normalize and combine
    merged["history_norm"] = min_max_normalize(merged["history_score"])
    merged["pref_norm"] = min_max_normalize(merged["pref_score"])

    merged["final_score"] = (
        alpha * merged["pref_norm"] + (1.0 - alpha) * merged["history_norm"]
    )

    # drop stuff with non positive final score
    merged = merged[merged["final_score"] > 0]

    # 6) foreign filter using TMDB language
    if effective_foreign_choice in ("exclude", "only"):
        langs = []
        for _, row in merged.iterrows():
            langs.append(get_original_language(row.get("tmdb_id")))
        merged["original_language"] = langs

        # if we failed to fetch any languages, skip foreign filtering
        if merged["original_language"].notna().sum() > 0:
            if effective_foreign_choice == "exclude":
                merged = merged[
                    merged["original_language"].isna()
                    | (merged["original_language"] == "en")
                ]
            else:  # "only"
                merged = merged[
                    merged["original_language"].notna()
                    & (merged["original_language"] != "en")
                ]

    # 7) hidden gems vs normal popularity
    if popularity_pref == "hidden":
        hidden_threshold = 5000  # tweak if you want

        merged = merged.sort_values("final_score", ascending=False)

        hidden = merged[merged["num_ratings"] < hidden_threshold]

        if len(hidden) >= top_n:
            merged = hidden.head(top_n)
        else:
            rest = merged[merged["num_ratings"] >= hidden_threshold]
            merged = pd.concat(
                [hidden, rest.head(top_n - len(hidden))],
                ignore_index=True,
            )
    else:
        merged = merged.sort_values("final_score", ascending=False).head(top_n)

    # 8) convert to list of dicts for the template
    results = []
    for _, row in merged.iterrows():
        year_val = row.get("year")
        year_out = int(year_val) if not pd.isna(year_val) else None

        results.append(
            {
                "movie_id": int(row["movie_id"]),
                "title": row["title"],
                "genres": row["genres"],
                "year": year_out,
                "avg_rating": float(row["avg_rating"]),
                "num_ratings": int(row["num_ratings"]),
                "history_score": float(row["history_score"]),
                "pref_score": float(row["pref_score"]),
                "final_score": float(row["final_score"]),
                "poster_url": get_poster_url(row.get("tmdb_id")),
            }
        )

    return results

# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    results = None

    # defaults
    default_genres = ["Horror"]
    default_year_choice = "any"
    default_pop_choice = "mixed"
    default_alpha = 0.4
    default_foreign_choice = "any"

    if request.method == "POST":
        try:
            user_id = MY_USER_ID
            genres_chosen = request.form.getlist("genres")

            year_choice = request.form.get("year_choice", "any")
            if year_choice == "after2000":
                year_min = 2000
            elif year_choice == "after1980":
                year_min = 1980
            else:
                year_min = 1900

            pop_choice = request.form.get("pop_choice", "mixed")
            alpha = float(request.form.get("alpha", "0.4"))
            foreign_choice = request.form.get("foreign_choice", "any")

            results = compute_hybrid_recs(
                user_id=user_id,
                genres_chosen=genres_chosen,
                year_min=year_min,
                popularity_pref=pop_choice,
                alpha=alpha,
                foreign_choice=foreign_choice,
                top_n=10,
            )

            default_genres = genres_chosen or default_genres
            default_year_choice = year_choice
            default_pop_choice = pop_choice
            default_alpha = alpha
            default_foreign_choice = foreign_choice

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        error=error,
        results=results,
        default_genres=default_genres,
        default_year_choice=default_year_choice,
        default_pop_choice=default_pop_choice,
        default_alpha=default_alpha,
        default_foreign_choice=default_foreign_choice,
    )


@app.route("/rate", methods=["POST"])
def rate():
    try:
        user_id = MY_USER_ID
        movie_id = int(request.form["movie_id"])
        rating = float(request.form["rating"])

        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO ratings (user_id, movie_id, rating, rating_ts)
                    VALUES (:uid, :mid, :r, NOW())
                    ON CONFLICT (user_id, movie_id)
                    DO UPDATE
                    SET rating = EXCLUDED.rating,
                        rating_ts = EXCLUDED.rating_ts
                """),
                {"uid": user_id, "mid": movie_id, "r": rating},
            )

        # AJAX case
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"ok": True, "movie_id": movie_id})

        # normal form fallback
        return redirect(url_for("index"))

    except Exception as e:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"ok": False, "error": str(e)}), 400
        return str(e), 400


if __name__ == "__main__":
    app.run(debug=True)
