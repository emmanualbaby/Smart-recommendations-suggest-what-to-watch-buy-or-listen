# pip install pandas scikit-learn numpy 

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Example product dataset
# -----------------------------
data = {
    'product': [
        'iPhone 15', 'AirPods Pro', 'MacBook Air', 
        'iPhone Case', 'Apple Watch', 'Wireless Charger',
        'Samsung Galaxy S24', 'Galaxy Buds', 'Smartwatch Charger'
    ],
    'description': [
        'Latest Apple smartphone with A17 chip and OLED display.',
        'Noise-cancelling wireless earbuds by Apple.',
        'Lightweight Apple laptop with M2 chip.',
        'Protective silicone case for iPhone.',
        'Smartwatch with fitness tracking and heart monitor.',
        'Fast wireless charger compatible with iPhone and Android.',
        'Newest Samsung flagship phone with great camera.',
        'Wireless earbuds with noise cancellation for Samsung devices.',
        'Charger compatible with all smartwatches.'
    ],
    'category': [
        'smartphone', 'earbuds', 'laptop', 
        'accessory', 'smartwatch', 'charger',
        'smartphone', 'earbuds', 'charger'
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# 2. Combine features into one text
# -----------------------------
df['features'] = df['description'] + " " + df['category']

# -----------------------------
# 3. Convert to vectors (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['features'])

# -----------------------------
# 4. Compute similarity between products
# -----------------------------
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# -----------------------------
# 5. Define recommender function
# -----------------------------
def recommend(product, num_recommendations=3):
    if product not in df['product'].values:
        return f"Sorry, '{product}' not found in product list."
    
    idx = df.index[df['product'] == product][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    
    recommendations = [df.iloc[i[0]]['product'] for i in sim_scores]
    return recommendations

# -----------------------------
# 6. Example usage
# -----------------------------
user_input = "iPhone 15"
print(f"Because you viewed '{user_input}', you might also like:")
for rec in recommend(user_input, 3):
    print(" -", rec)
