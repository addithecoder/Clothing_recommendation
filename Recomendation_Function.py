import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_recomended_items(input_text):
    # Load the database
    database_file = "clothing_data2.csv"
    df = pd.read_csv(database_file)
    top_n = 3

    # Preprocess the data, if needed

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the item descriptions in the database
    tfidf_matrix = vectorizer.fit_transform(df['Description'])

    # Convert the input text to a TF-IDF vector
    input_vector = vectorizer.transform([input_text])

    # Compute the cosine similarity between the input vector and all items
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)

    # Get the indices of the most similar items
    top_indices = similarity_scores.argsort()[0][::-1][:top_n]

    # Retrieve the URLs of the top similar items
    top_urls = df.iloc[top_indices]['Url'].tolist()

    return top_urls