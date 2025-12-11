import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.metrics.pairwise import cosine_similarity

# Shared DB engine
engine = create_engine(
    "postgresql+psycopg2://tylerbrecker@localhost:5433/recsys_db"
)


def _build_local_matrix(
    user_id: int,
    max_neighbors: int = 3000,
    min_movie_ratings: int = 5,
):
    """
    Build a ratings matrix only for:
      - the target user
      - a limited set of "neighbor" users who rated at least one of the same movies
    This keeps the pivot small so we do not blow up memory.
    """
    # Ratings for this user
    user_ratings = pd.read_sql(
    "SELECT movie_id, rating FROM ratings WHERE user_id = %s",
    engine,
    params=(user_id,),   # single-element tuple
    )
    
    if user_ratings.empty:
        raise ValueError(f"user_id {user_id} has no ratings")

    rated_ids = user_ratings["movie_id"].tolist()

    # Find neighbor users who rated at least one of those movies
    placeholders = ",".join(["%s"] * len(rated_ids))
    neighbor_df = pd.read_sql(
        f"""
        SELECT DISTINCT user_id
        FROM ratings
        WHERE movie_id IN ({placeholders})
          AND user_id <> %s
        LIMIT {max_neighbors}
        """,
        engine,
        params=tuple(rated_ids + [user_id]),   # <<< change here
    )

    neighbor_ids = neighbor_df["user_id"].tolist()
    all_users = neighbor_ids + [user_id]

    if not all_users:
        # Only this user has ratings for those movies, fall back to just them
        all_users = [user_id]

    # Pull all ratings for this small user set
    placeholders_users = ",".join(["%s"] * len(all_users))
    ratings_local = pd.read_sql(
        f"""
        SELECT user_id, movie_id, rating
        FROM ratings
        WHERE user_id IN ({placeholders_users})
        """,
        engine,
        params=tuple(all_users),              # <<< and here
    )

    # Keep only movies that have enough ratings within this subset
    movie_counts = ratings_local["movie_id"].value_counts()
    popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
    ratings_local = ratings_local[ratings_local["movie_id"].isin(popular_movies)]

    if ratings_local.empty:
        raise ValueError("Not enough overlapping data to build local matrix")

    rating_matrix = ratings_local.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
    )

    rating_matrix_filled = rating_matrix.fillna(0)
    return rating_matrix, rating_matrix_filled


def recommend_from_history(
    user_id: int,
    top_n: int = 10,
    min_ratings: int = 30,
) -> pd.DataFrame:
    """
    Recommend movies based only on this user's rating history, using
    a local item-item similarity matrix built around their neighborhood.
    """
    # 1. Build local rating matrix
    rating_matrix, rating_matrix_filled = _build_local_matrix(user_id,)

    movie_ids = rating_matrix_filled.columns

    # 2. Item item similarity within this local subset
    sim_matrix = cosine_similarity(rating_matrix_filled.T)
    sim_df = pd.DataFrame(sim_matrix, index=movie_ids, columns=movie_ids)

    # 3. User's own ratings row (with NaN for unrated movies)
    user_ratings = rating_matrix.loc[user_id]
    rated = user_ratings.dropna()
    if rated.empty:
        raise ValueError(f"user_id {user_id} has no ratings")

    rated_movie_ids = rated.index.tolist()

    # Start scores at zero for all movies in this local set
    scores = pd.Series(0.0, index=movie_ids)

    # Spread preference from each rated movie to its similar neighbors
    for movie_id, rating in rated.items():
        rating_delta = rating - 3.0  # 3 is neutral
        if movie_id not in sim_df.columns:
            continue
        scores += sim_df[movie_id] * rating_delta

    # Drop movies already seen
    scores = scores.drop(labels=rated_movie_ids, errors="ignore")

    # Keep only positive scores
    score_df = scores.to_frame("score")
    score_df = score_df[score_df["score"] > 0]

    if score_df.empty:
        raise ValueError("No positive scored movies found for this user")

    # 4. Join with movie metadata
    movie_stats = pd.read_sql(
        """
        SELECT movie_id, title, genres, avg_rating, num_ratings
        FROM movie_features_with_links
        """,
        engine,
    ).set_index("movie_id")

    score_df = score_df.join(movie_stats, how="inner")

    # Filter out movies with very few ratings in the full dataset
    score_df = score_df[score_df["num_ratings"] >= min_ratings]

    if score_df.empty:
        raise ValueError("No candidates left after min_ratings filter")

    # Sort and take top_n
    score_df = score_df.sort_values("score", ascending=False).head(top_n)
    score_df = score_df.reset_index().rename(columns={"index": "movie_id"})

    cols = ["movie_id", "title", "genres", "score", "avg_rating", "num_ratings"]
    return score_df[cols]


if __name__ == "__main__":
    # Manual quick test, does not run when imported by Flask
    for uid in [1, 10, 50]:
        print(f"\nRecommendations for user {uid}:")
        try:
            recs = recommend_from_history(uid, top_n=10)
            print(recs.to_string(index=False))
        except ValueError as e:
            print("  ", e)
