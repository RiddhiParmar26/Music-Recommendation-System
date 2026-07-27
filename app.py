from flask import Flask, render_template, request, session, redirect
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "music_ai_secret"

# Load Dataset
df = pd.read_csv("music.csv")

# Create Features
df["Features"] = df["Artist"] + " " + df["Genre"] + " " + df["Mood"]

# TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Features"])

# Cosine Similarity
similarity = cosine_similarity(tfidf_matrix)


@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []

    if request.method == "POST":

        selected_song = request.form.get("song")
        mood = request.form.get("mood")

        added_songs = set()

        # -------------------------------
        # Mood Recommendations
        # -------------------------------

        if mood:

            mood_songs = df[df["Mood"] == mood]

            for _, row in mood_songs.iterrows():

                if row["Song"] != selected_song and row["Song"] not in added_songs:

                    recommendations.append({
                        "song": row["Song"],
                        "artist": row["Artist"],
                        "genre": row["Genre"],
                        "mood": row["Mood"],
                        "image": row["Image"],
                        "link": row["Link"],
                        "score": 100
                    })

                    added_songs.add(row["Song"])

                if len(recommendations) >= 5:
                    break

        # -------------------------------
        # Similar Song Recommendations
        # -------------------------------

        if selected_song:

            index = df[df["Song"] == selected_song].index

            if len(index):

                index = index[0]

                scores = list(enumerate(similarity[index]))
                scores = sorted(scores, key=lambda x: x[1], reverse=True)

                for i in scores[1:]:

                    row = df.iloc[i[0]]

                    if row["Song"] not in added_songs:

                        recommendations.append({
                            "song": row["Song"],
                            "artist": row["Artist"],
                            "genre": row["Genre"],
                            "mood": row["Mood"],
                            "image": row["Image"],
                            "link": row["Link"],
                            "score": round(i[1] * 100)
                        })

                        added_songs.add(row["Song"])

                    if len(recommendations) >= 5:
                        break

    return render_template(
        "index.html",
        songs=df["Song"].tolist(),
        moods=df["Mood"].unique().tolist(),
        recommendations=recommendations
    )

@app.route("/favorites")
def favorites():

    fav_songs = session.get("favorites", [])

    return render_template(
        "favorites.html",
        favorites=fav_songs
    )



@app.route("/add_favorite", methods=["POST"])
def add_favorite():

    song = request.form.get("song")

    favorites = session.get("favorites", [])

    if song not in favorites:
        favorites.append(song)

    session["favorites"] = favorites

    return redirect("/")
               

import os

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )