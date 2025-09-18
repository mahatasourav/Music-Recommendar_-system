from flask import Flask, request, jsonify
import pickle
import difflib

app = Flask(__name__)

# ----------------------------
# Load saved model + vectorizer + data
# ----------------------------
with open("nn_model.pkl", "rb") as f:
    nn_model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("songs.pkl", "rb") as f:
    df = pickle.load(f)

# ----------------------------
# Recommendation function
# ----------------------------
def recommend_songs(song_name, top_n=5):
    # Lowercase list of songs
    all_songs = df['song'].str.lower().tolist()

    # Find closest match
    close_matches = difflib.get_close_matches(song_name.lower(), all_songs, n=1, cutoff=0.4)
    if not close_matches:
        return []  # No match found

    matched_song = close_matches[0]
    idx = df[df['song'].str.lower() == matched_song].index[0]

    # Nearest neighbors
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=top_n+1)

    recs = []
    for i in indices[0][1:]:  # skip itself
        recs.append({
            "song": df.iloc[i]['song'],
            "artist": df.iloc[i]['artist']
        })
    return recs

# ----------------------------
# Flask route
# ----------------------------
@app.route("/recommend", methods=["GET"])
def recommend_api():
    song_name = request.args.get("song")
    if not song_name:
        return jsonify({"error": "Please provide a song name using ?song=SongName"}), 400
    
    recommendations = recommend_songs(song_name)
    
    if not recommendations:
        return jsonify({"error": f"No matches found for '{song_name}'"}), 404

    return jsonify({"recommendations": recommendations})

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
