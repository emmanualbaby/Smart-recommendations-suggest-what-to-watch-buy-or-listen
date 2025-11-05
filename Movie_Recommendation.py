# Install dependencies if needed
# pip install pandas scikit-learn numpy

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Example dataset
# -----------------------------
data = {
    'title': [
        'Breaking Bad', 'Money Heist', 'Stranger Things',
        'Narcos', 'Dark', 'Ozark', 'Peaky Blinders', 'The Witcher'
    ],
    'description': [
        'A chemistry teacher turns to making meth to secure his family’s future.',
        'A group of robbers plan the biggest heist in Spanish history.',
        'Kids uncover a supernatural mystery in a small town.',
        'A drug kingpin’s rise and fall in Colombia.',
        'Time travel and dark family secrets in a small German town.',
        'A financial advisor becomes a money launderer for a cartel.',
        'A gangster family in post-war Birmingham.',
        'A monster hunter struggles between humans and beasts.'
    ],
    'genre': [
        'crime drama', 'thriller heist', 'sci-fi mystery',
        'crime drama', 'sci-fi mystery', 'crime drama',
        'crime drama', 'fantasy action'
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Combine text features
# -----------------------------
df['features'] = df['description'] + " " + df['genre']

# -----------------------------
# 3. Convert text to vectors (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['features'])

# -----------------------------
# 4. Compute similarity
# -----------------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -----------------------------
# 5. Recommendation function
# -----------------------------
def recommend(title, num_recommendations=3):
    if title not in df['title'].values:
        return f"Sorry, '{title}' not found in database."
    
    idx = df.index[df['title'] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    recommendations = [df.iloc[i[0]]['title'] for i in sim_scores]
    return recommendations

# -----------------------------
# 6. Example usage
# -----------------------------
user_input = "Breaking Bad"
print(f"Because you liked '{user_input}', you might enjoy:")
for rec in recommend(user_input, 3):
    print(" -", rec)
