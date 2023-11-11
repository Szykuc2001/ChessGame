#!/usr/bin/env python
# coding: utf-8

# ## Movie Recommendation System

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval

# In[2]:


path = "./Users/Simon/PycharmProjects/ChessGame/Lab3"
credits_df = pd.read_csv(r"C:\Users\Simon\PycharmProjects\ChessGame\Lab3\tmdb_5000_credits.csv")
movies_df = pd.read_csv(r"C:\Users\Simon\PycharmProjects\ChessGame\Lab3\tmdb_5000_movies.csv")

# In[3]:


movies_df.head()

# In[4]:


credits_df.head()

# In[5]:


credits_df.columns = ['id', 'tittle', 'cast', 'crew']
movies_df = movies_df.merge(credits_df, on="id")

# In[6]:


movies_df.head()

# In[7]:


# Demographic Filtering
C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.9)

print("C: ", C)
print("m: ", m)

new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]
print(new_movies_df.shape)


# In[8]:


def weighted_rating(x, C=C, m=m):
    v = x["vote_count"]
    R = x["vote_average"]

    return (v / (v + m) * R) + (m / (v + m) * C)


# In[9]:


new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
new_movies_df = new_movies_df.sort_values('score', ascending=False)

new_movies_df[["title", "vote_count", "vote_average", "score"]].head(10)


# In[10]:


# Plot top 10 movies
def plot():
    popularity = movies_df.sort_values("popularity", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.barh(popularity["title"].head(10), popularity["popularity"].head(10), align="center", color="skyblue")
    plt.gca().invert_yaxis()
    plt.title("Top 10 movies")
    plt.xlabel("Popularity")
    plt.show()


plot()

# In[11]:


# Content based Filtering
print(movies_df["overview"].head(5))

# In[12]:


tfidf = TfidfVectorizer(stop_words="english")
movies_df["overview"] = movies_df["overview"].fillna("")

tfidf_matrix = tfidf.fit_transform(movies_df["overview"])
print(tfidf_matrix.shape)

# In[13]:


# Compute similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape)

indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()
print(indices.head())


# In[14]:


def get_recommendations(title, cosine_sim=cosine_sim):
    """
    in this function,
        we take the cosine score of given movie
        sort them based on cosine score (movie_id, cosine_score)
        take the next 10 values because the first entry is itself
        get those movie indices
        map those indices to titles
        return title list
    """
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    # (a, b) where a is id of movie, b is sim_score

    movies_indices = [ind[0] for ind in sim_scores]
    movies = movies_df["title"].iloc[movies_indices]
    return movies


# In[15]:


print("################ Content Based Filtering - plot#############")
print()
print("Recommendations for The Lord of the Rings: The Fellowship of the Ring")
print(get_recommendations("The Lord of the Rings: The Fellowship of the Ring"))
print()
print("Recommendations for Avengers")
print(get_recommendations("The Avengers"))
print()
print("Recommendations for The Godfather")
print(get_recommendations("The Godfather"))
print()
print("Recommendations for Inception")
print(get_recommendations("Inception"))
print()
print("Recommendations for The Matrix")
print(get_recommendations("The Matrix"))
print()
print("Recommendations for Interstellar")
print(get_recommendations("Interstellar"))
print()
print("Recommendations for Snowpiercer")
print(get_recommendations("Snowpiercer"))
print()
print("Recommendations for Return of the Jedi")
print(get_recommendations("Return of the Jedi"))
print()
print("Recommendations for Django Unchained")
print(get_recommendations("Django Unchained"))
print()
print("Recommendations for Shrek")
print(get_recommendations("Shrek"))
print()
print("Recommendations for Pacific Rim")
print(get_recommendations("Pacific Rim"))
print()
print("Recommendations for Titanic")
print(get_recommendations("Titanic"))
print()
print("Recommendations for Jurassic Park")
print(get_recommendations("Jurassic Park"))
print()
print("Recommendations for I, Robot")
print(get_recommendations("I, Robot"))
print()
print("Recommendations for Legend")
print(get_recommendations("Legend"))
print()
print("Recommendations for Raiders of the Lost Ark")
print(get_recommendations("Raiders of the Lost Ark"))

# In[16]:


features = ["cast", "crew", "keywords", "genres"]

for feature in features:
    movies_df[feature] = movies_df[feature].apply(literal_eval)

movies_df[features].head(10)


# In[17]:


def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan


# In[18]:


def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]

        if len(names) > 3:
            names = names[:3]

        return names

    return []


# In[19]:


movies_df["director"] = movies_df["crew"].apply(get_director)

features = ["cast", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(get_list)

# In[21]:


movies_df[['title', 'cast', 'director', 'keywords', 'genres']].head()


# In[22]:


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""


# In[23]:


features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)


# In[24]:


def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


movies_df["soup"] = movies_df.apply(create_soup, axis=1)
print(movies_df["soup"].head())

# In[25]:


count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(movies_df["soup"])

print(count_matrix.shape)

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim2.shape)

movies_df = movies_df.reset_index()
indices = pd.Series(movies_df.index, index=movies_df['title'])

# In[26]:


print("################ Content Based System - metadata #############")
print("Recommendations for The Dark Knight Rises")
print(get_recommendations("The Dark Knight Rises", cosine_sim2))
print()
print("Recommendations for Avengers")
print(get_recommendations("The Avengers", cosine_sim2))

# In[ ]: