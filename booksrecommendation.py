import pandas as pd 
import numpy as np
# print (scipy.__version__)
from IPython.display import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from flask import Flask, request, jsonify

app = Flask(__name__)

data = pd.read_csv('bestsellers_with_categories_2022_03_27.csv')

# riottenere dati quantitativi e qualitativi
data['Genre'] = np.where(data['Genre'] == 1, "Fiction", "Non Fiction")
quantitative_data = data[['Reviews','Year','Price']]
qualitative_data = data[['Name','Author','Genre']]

# Create a TF-IDF vectorizer for text features (Author and Genre)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Create a copy of the DataFrame to avoid SettingWithCopyWarning
qualitative_data = qualitative_data.copy()

# Combine text features into a single column
qualitative_data['TextFeatures'] = qualitative_data['Name'] + ' ' + qualitative_data['Author'] + ' ' + qualitative_data['Genre']

# Apply TF-IDF vectorization to text features
tfidf_matrix = tfidf_vectorizer.fit_transform(qualitative_data['TextFeatures'])

# Calculate cosine similarities between books based on text features
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend books based on content similarity with unique book names
def get_unique_recommendations(book_title, num_recommendations=10, cosine_sim=cosine_sim):
    idx = qualitative_data[qualitative_data['Name'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:100]  # Get the top 100 most similar books (excluding itself)
    book_indices = [i[0] for i in sim_scores]

    # Create a set to keep track of unique book names
    unique_books = set()
    unique_recommendations = []

    for idx in book_indices:
        book_name = qualitative_data['Name'].iloc[idx]
        if book_name not in unique_books and book_name != book_title:
            unique_books.add(book_name)
            unique_recommendations.append(book_name)

            if len(unique_recommendations) == num_recommendations:
                break

    return unique_recommendations

# Define an API endpoint for book recommendations
@app.route('/get_recommendations', methods=['GET'])
def recommend_books():
    book_title = request.args.get('title')
    num_recommendations = int(request.args.get('num_recommendations', 10))

    recommended_books = get_unique_recommendations(book_title, num_recommendations)
    
    return jsonify({"recommendations": recommended_books})

if __name__ == '__main__':
    app.run(debug=True)