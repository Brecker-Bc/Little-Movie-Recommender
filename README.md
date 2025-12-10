🎬 Hybrid Movie Recommender System

A fully interactive movie recommendation platform built with Flask, PostgreSQL, and pandas.
It blends collaborative filtering with real-time preference scoring to deliver flexible, personalized recommendations through a clean web interface.

🔧 Key Features
✔ Hybrid Recommendation Engine

The system merges two scoring strategies:

History score
Identifies movies similar to ones the user has rated highly.

Preference score
Responds to choices made in the UI, including genres, release eras, popularity ranges, and optional foreign-film filtering.

A user-controlled slider combines them into a final weighted score.

✔ Automatic User Profiles

Each device is treated as a unique user.
All ratings are stored in PostgreSQL and immediately influence future recommendations.

✔ Interactive Rating UI

Movie cards include 1 to 5 star rating buttons.
Rating a film removes it from future suggestion pools and updates collaborative filtering results on the fly.

✔ Responsive Front End

A Bootstrap layout renders movie cards with metadata, genres, and optional poster images pulled from the TMDB API.

✔ Rich PostgreSQL-Backed Dataset

The system enriches the MovieLens dataset with:

genre indicators

rating counts and averages

optional TMDB metadata for posters and popularity

✔ Foreign-Film Controls

Users can filter by:

all films

English-only

foreign-language only

🧠 How the Hybrid Score Works

Each unrated movie receives two normalized scores:

history_norm for collaborative similarity

pref_norm for preference alignment

They blend into the final ranking:

final_score = α * pref_norm + (1 − α) * history_norm


Adjusting α shifts recommendations between
“movies similar to my history” and
“movies that fit my preferences right now.”

📦 Tech Stack

Python

Flask

PostgreSQL + SQLAlchemy

Pandas / NumPy

Bootstrap
