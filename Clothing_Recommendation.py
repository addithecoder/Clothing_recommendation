#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


# In[2]:


df = pd.read_csv('clothing_data.csv')


# In[3]:


df


# In[5]:


df.isnull()


# In[6]:


df.isnull().sum()


# In[7]:


df.dropna(inplace=True)


# In[8]:


df.isnull().sum()


# In[12]:


def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and punctuation
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# In[13]:


df['Description'] = df['Description'].apply(clean_text)


# In[14]:


vectorizer = TfidfVectorizer(stop_words='english')


# In[16]:


tfidf_matrix = vectorizer.fit_transform(df['Description'])


# In[20]:


def compute_similarity(input_text):
    # Convert the input text to a TF-IDF vector
    input_vector = vectorizer.transform([input_text])

    # Compute the cosine similarity between the input vector and all items
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)

    # Get the indices and similarity scores of the most similar items
    top_items = similarity_scores.argsort()[0][::-1]

    # Retrieve the indices and similarity scores of the top items
    item_indices = top_items[:5]  # Get the top 5 most similar items
    item_scores = similarity_scores[0, item_indices]

    # Return the most similar items and their similarity scores
    recommendations = df.iloc[item_indices][['ProductID', 'Description']]
    recommendations['similarity_score'] = item_scores

    return recommendations


# In[36]:


input_text = "male shoes"


# In[37]:


similar_items = compute_similarity(input_text)


# In[38]:


print(similar_items)


# In[49]:


def get_similar_items(input_text, database_file, top_n=5):
    # Load the database
    df = pd.read_csv(database_file)
    
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


# In[60]:


input_text2 = "Black cotton t-shirt"


# In[61]:


database_file = "clothing_data2.csv"


# In[62]:


#Printing top 3 results
top_n = 3


# In[63]:


similar_items2 = get_similar_items(input_text2, database_file, top_n)


# In[59]:


print(similar_items2)
#Urls are not present in the database, so the function therefore returns the Name of the product as per the dataset used.


# In[ ]:




