import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle

# Step 1: Load dataset
print("Loading dataset...")
df = pd.read_csv("spotify_millsongdata.csv")

# Step 2: TF-IDF Vectorization
print("Building TF-IDF matrix...")
tfidf = TfidfVectorizer(stop_words='english', max_features=20000)  # limit features to save memory
tfidf_matrix = tfidf.fit_transform(df['text'])

# Step 3: Train Nearest Neighbors model
print("Training Nearest Neighbors model...")
nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
nn_model.fit(tfidf_matrix)

# Step 4: Save vectorizer, matrix, and model
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f)

with open("nn_model.pkl", "wb") as f:
    pickle.dump(nn_model, f)

with open("songs.pkl", "wb") as f:
    pickle.dump(df[['artist', 'song']], f)

print("✅ Training complete! Files saved: tfidf_vectorizer.pkl, tfidf_matrix.pkl, nn_model.pkl, songs.pkl")
