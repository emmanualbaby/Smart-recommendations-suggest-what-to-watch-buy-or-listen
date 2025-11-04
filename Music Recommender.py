# pip install pandas scikit-learn numpy

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# -----------------------------
# 1. Example music dataset
# -----------------------------
songs = pd.DataFrame({
    'title': [
        'Blinding Lights', 'Save Your Tears', 'Levitating',
        'Watermelon Sugar', 'Peaches', 'As It Was', 'Stay', 'Heat Waves'
    ],
    'artist': [
        'The Weeknd', 'The Weeknd', 'Dua Lipa',
        'Harry Styles', 'Justin Bieber', 'Harry Styles', 'The Kid LAROI', 'Glass Animals'
    ],
    # Simplified audio features (Spotify provides many: tempo, energy, valence, danceability, etc.)
    'danceability': [0.8, 0.75, 0.79, 0.65, 0.72, 0.6, 0.85, 0.74],
    'energy': [0.73, 0.68, 0.82, 0.58, 0.65, 0.55, 0.88, 0.7],
    'valence': [0.8, 0.6, 0.9, 0.75, 0.7, 0.65, 0.75, 0.6],
    'tempo': [171, 118, 103, 95, 90, 86, 170, 92]
})

# -----------------------------
# 2. Normalize feature values
# -----------------------------
scaler = StandardScaler()
features = scaler.fit_transform(songs[['danceability', 'energy', 'valence', 'tempo']])

# -----------------------------
# 3. Compute cosine similarity between songs
# -----------------------------
similarity = cosine_similarity(features, features)

# -----------------------------
# 4. Recommender function
# -----------------------------
def recommend(song_title, num=3):
    if song_title not in songs['title'].values:
        return f"Sorry, '{song_title}' not found."
    
    idx = songs.index[songs['title'] == song_title][0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num+1]
    
    recs = [(songs.iloc[i[0]]['title'], songs.iloc[i[0]]['artist']) for i in sim_scores]
    return recs

# -----------------------------
# 5. Example usage
# -----------------------------
song_input = "Blinding Lights"
print(f"Because you liked '{song_input}', you might enjoy:")
for rec in recommend(song_input, 3):
    print(f" - {rec[0]} by {rec[1]}")
