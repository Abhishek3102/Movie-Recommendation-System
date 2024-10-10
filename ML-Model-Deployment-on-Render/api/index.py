from flask import Flask, request, jsonify
import pickle
import difflib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

model_path = 'movies_model.pkl'
with open(model_path, 'rb') as file:
    df = pickle.load(file)

selected_features = ["genres", "keywords", "tagline", "cast", "director"]
for feature in selected_features:
    df[feature] = df[feature].fillna("")
combined_features = df["genres"] + " " + df["keywords"] + " " + df["tagline"] + " " + df["cast"] + " " + df["director"]
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.json.get('movie_name', '')
    all_movie_names = df['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, all_movie_names)

    if not find_close_match:
        return jsonify({"error": "No match found. Please try again."}), 400

    close_match = find_close_match[0]

    try:
        movie_index = df[df.title == close_match].index[0]
    except IndexError:
        return jsonify({"error": "Error finding movie index. Please try again."}), 400

    try:
        similarity_score = list(enumerate(similarity[movie_index]))
    except IndexError:
        return jsonify({"error": "Error accessing similarity scores. Please try again."}), 400

    sorted_similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    for i, movie in enumerate(sorted_similarity_score[1:31]):  
        index = movie[0]
        try:
            title_from_index = df.loc[index, 'title']
        except KeyError:
            continue
        recommended_movies.append(title_from_index)

    return jsonify({"recommended_movies": recommended_movies})

if __name__ == "__main__":
    app.run(debug=True)
