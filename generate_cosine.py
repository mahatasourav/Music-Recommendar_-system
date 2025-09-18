import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# CSV load
df = pd.read_csv("spotify_millsongdata.csv")

# TF-IDF vectorization of lyrics
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['text'])

# Cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Save cosine similarity to pickle
with open("cosine_sim.pkl", "wb") as f:
    pickle.dump(cosine_sim, f)

print("cosine_sim.pkl generated successfully!")
