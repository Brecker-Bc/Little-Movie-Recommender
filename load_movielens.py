import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path

# Base/path setup
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"   # remove "data" if your CSVs are directly next to the script

# 1. Read CSVs
movies = pd.read_csv(DATA_DIR / "movies.csv")
ratings = pd.read_csv(DATA_DIR / "ratings.csv")
tags = pd.read_csv(DATA_DIR / "tags.csv")
links = pd.read_csv(DATA_DIR / "links.csv")

movies.columns = movies.columns.str.lower()
ratings.columns = ratings.columns.str.lower()
tags.columns = tags.columns.str.lower()
links.columns = links.columns.str.lower()

# 2. Build users table from ratings and tags
user_ids = pd.unique(pd.concat([ratings["userid"], tags["userid"]], ignore_index=True))
users = pd.DataFrame({"user_id": user_ids})

# 3. Connect to Postgres
engine = create_engine("postgresql+psycopg2://tylerbrecker@localhost:5433/recsys_db")

with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE ratings, tags, links, movies, users RESTART IDENTITY"))

# 4. Insert users and movies
users.to_sql("users", engine, if_exists="append", index=False)

movies = movies.rename(columns={"movieid": "movie_id"})
movies.to_sql("movies", engine, if_exists="append", index=False)

# 5. Insert ratings 
ratings = ratings.rename(columns={"userid": "user_id", "movieid": "movie_id", "timestamp": "rating_ts"})
ratings["rating_ts"] = pd.to_datetime(ratings["rating_ts"], unit="s")
ratings.to_sql("ratings", engine, if_exists="append", index=False)

# 6. Insert tags 
tags = tags.rename(columns={"userid": "user_id", "movieid": "movie_id", "timestamp": "tag_ts"})
tags["tag_ts"] = pd.to_datetime(tags["tag_ts"], unit="s")
tags.to_sql("tags", engine, if_exists="append", index=False)

# 7. Insert links
links = links.rename(columns={
    "movieid": "movie_id",
    "imdbid": "imdb_id",
    "tmdbid": "tmdb_id"
})
links.to_sql("links", engine, if_exists="append", index=False)

