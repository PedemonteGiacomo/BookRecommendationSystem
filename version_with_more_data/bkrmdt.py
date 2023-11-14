import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix

# Read your CSV file
data = pd.read_csv('books_data/books.csv', sep=";", error_bad_lines=False, encoding="latin-1")

# Drop or fill missing values in relevant columns
data = data.dropna(subset=['Book-Title', 'Book-Author', 'Publisher'])

data_qualitative = data[["Book-Title","Book-Author","Publisher"]]

# Sample a smaller subset for development
sampled_data = data_qualitative.sample(frac=0.1, random_state=42)

# Combine text features into a single column
sampled_data['TextFeatures'] = sampled_data['Book-Title'] + ' ' + sampled_data['Book-Author']

# Create a TF-IDF vectorizer for text features (Author and Genre)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Apply TF-IDF vectorization to text features
tfidf_matrix = tfidf_vectorizer.fit_transform(sampled_data['TextFeatures'])

# Convert tfidf_matrix to a sparse matrix
tfidf_sparse = tfidf_matrix.tocsr()

# Calculate cosine similarities between books based on text features
cosine_sim = linear_kernel(tfidf_sparse, tfidf_sparse)

# Function to recommend books based on content similarity with unique book names
def get_unique_recommendations(book_title, author, num_recommendations=10):
    global sampled_data

    if book_title not in sampled_data['Book-Title'].values:
        print(f"Warning: Book with title '{book_title}' not found in the dataset. Recommendations are based on available data.")
        # Add the new book to the dataset
        new_data = {'Book-Title': [book_title], 'Book-Author': [author]}
        sampled_data = sampled_data.append(pd.DataFrame(new_data), ignore_index=True)

        # Update TF-IDF matrix and cosine similarity matrix
        new_text_feature = f"{book_title} {author}"
        new_tfidf_vector = tfidf_vectorizer.transform([new_text_feature])
        tfidf_sparse.resize((tfidf_sparse.shape[0] + 1, tfidf_sparse.shape[1]))
        tfidf_sparse[-1] = new_tfidf_vector
        cosine_sim.resize((cosine_sim.shape[0] + 1, cosine_sim.shape[1] + 1))
        cosine_sim[-1] = linear_kernel(new_tfidf_vector, tfidf_sparse).flatten()

    idx = sampled_data[sampled_data['Book-Title'] == book_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort and get the top 100 most similar books (excluding itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:100]

    # Create a set to keep track of unique book names
    unique_books = set()
    unique_recommendations = []

    for idx in [i[0] for i in sim_scores]:
        book_name = sampled_data['Book-Title'].iloc[idx]
        if book_name not in unique_books and book_name != book_title:
            unique_books.add(book_name)
            unique_recommendations.append(book_name)

            if len(unique_recommendations) == num_recommendations:
                break

    return unique_recommendations

# Example usage
recommendations = get_unique_recommendations(book_title="Percy Jackson", author="Rick Riordan")
print(recommendations)