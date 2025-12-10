import pandas as pd
from sqlalchemy import create_engine, text

# Adjust if your connection string is different
engine = create_engine(
    "postgresql+psycopg2://tylerbrecker@localhost:5433/recsys_db"
)

DATA_DIR = "data"  # relative path to the unzipped folder


def create_tables():
    ddl = """
    DROP TABLE IF EXISTS ratings CASCADE;
    DROP TABLE IF EXISTS tags CASCADE;
    DROP TABLE IF EXISTS links CASCADE;
    DROP TABLE IF EXISTS movies CASCADE;
    DROP TABLE IF EXISTS users CASCADE;

    CREATE TABLE movies (
        movie_id   INT PRIMARY KEY,
        title      TEXT,
        genres     TEXT
    );

    CREATE TABLE links (
        movie_id INT PRIMARY KEY,
        imdb_id  INT,
        tmdb_id  INT
    );

    CREATE TABLE users (
        user_id INT PRIMARY KEY
    );

    CREATE TABLE ratings (
        user_id    INT REFERENCES users(user_id),
        movie_id   INT REFERENCES movies(movie_id),
        rating     NUMERIC(2,1),
        rating_ts  TIMESTAMP
    );

    CREATE TABLE tags (
        user_id   INT REFERENCES users(user_id),
        movie_id  INT REFERENCES movies(movie_id),
        tag       TEXT,
        tag_ts    TIMESTAMP
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    print("Recreated movies, links, users, ratings, tags tables.")

def load_movies_and_links():
    movies_path = f"{DATA_DIR}/movies.csv"
    links_path = f"{DATA_DIR}/links.csv"

    movies = pd.read_csv(movies_path)
    links = pd.read_csv(links_path)

    # Rename columns to match our SQL schema
    movies = movies.rename(columns={"movieId": "movie_id"})
    links = links.rename(
        columns={
            "movieId": "movie_id",
            "imdbId": "imdb_id",
            "tmdbId": "tmdb_id",
        }
    )

    # Clean up types a bit
    movies["title"] = movies["title"].astype(str)
    movies["genres"] = movies["genres"].astype(str)

    # tmdbId has blanks; make them numeric with NaN allowed
    links["tmdb_id"] = pd.to_numeric(links["tmdb_id"], errors="coerce")
    links["imdb_id"] = pd.to_numeric(links["imdb_id"], errors="coerce")

    with engine.begin() as conn:
        movies.to_sql("movies", conn, if_exists="append", index=False)
        links.to_sql("links", conn, if_exists="append", index=False)

    print(f"Loaded {len(movies)} movies and {len(links)} links.")


def load_ratings_in_chunks(chunksize=1_000_000):
    ratings_path = f"{DATA_DIR}/ratings.csv"

    # ---------- First pass: create users ----------
    # Read only the userId column once, get unique users
    print("Collecting users from ratings...")
    user_ids = pd.read_csv(ratings_path, usecols=["userId"])["userId"].unique()
    users = pd.DataFrame({"user_id": user_ids})

    with engine.begin() as conn:
        users.to_sql("users", conn, if_exists="append", index=False)

    print(f"Loaded {len(users)} users.")

    # ---------- Second pass: stream ratings ----------
    print("Streaming ratings into database...")
    chunk_iter = pd.read_csv(ratings_path, chunksize=chunksize)

    total_rows = 0
    for i, chunk in enumerate(chunk_iter, start=1):
        # convert unix timestamp to real timestamp
        chunk["rating_ts"] = pd.to_datetime(chunk["timestamp"], unit="s")
        chunk = chunk.drop(columns=["timestamp"])

        # rename to match DB columns
        chunk = chunk.rename(
            columns={
                "userId": "user_id",
                "movieId": "movie_id",
                "rating": "rating",
            }
        )

        with engine.begin() as conn:
            chunk.to_sql("ratings", conn, if_exists="append", index=False)

        total_rows += len(chunk)
        print(f"Inserted ratings chunk {i}, rows so far {total_rows}")

    print(f"Loaded {total_rows} ratings.")

def load_tags_in_chunks(chunksize=500_000):
    tags_path = f"{DATA_DIR}/tags.csv"
    chunk_iter = pd.read_csv(tags_path, chunksize=chunksize)

    total_rows = 0

    for i, chunk in enumerate(chunk_iter, start=1):
        chunk["tag_ts"] = pd.to_datetime(chunk["timestamp"], unit="s")
        chunk = chunk.drop(columns=["timestamp"])

        chunk = chunk.rename(
            columns={
                "userId": "user_id",
                "movieId": "movie_id",
                "tag": "tag",
            }
        )

        with engine.begin() as conn:
            chunk.to_sql("tags", conn, if_exists="append", index=False)

        total_rows += len(chunk)
        print(f"Inserted tags chunk {i}, rows so far {total_rows}")

    print(f"Loaded {total_rows} tags.")


def main():
    create_tables()
    load_movies_and_links()
    load_ratings_in_chunks()
    load_tags_in_chunks()
    print("Finished loading MovieLens 25M.")


if __name__ == "__main__":
    main()
