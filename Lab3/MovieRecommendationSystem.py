#Instructions
'''
1. Download required files
2. Put credits and movies files in the project directory
3. Change path for the files to their absolute path
4. Change the movie titles in the appropriate lines
5. Run the code
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval


'''Reading CSV files into pandas DataFrames'''
credits_df = pd.read_csv(r"C:\Users\Simon\PycharmProjects\ChessGame\Lab3\tmdb_5000_credits.csv")
movies_df = pd.read_csv(r"C:\Users\Simon\PycharmProjects\ChessGame\Lab3\tmdb_5000_movies.csv")


movies_df.head()


credits_df.head()

'''Renaming columns in the credits DataFrame'''
credits_df.columns = ['id', 'tittle', 'cast', 'crew']
'''Merging movies and credits DataFrames on 'id' column'''
movies_df = movies_df.merge(credits_df, on="id")


movies_df.head()


# Demographic Filtering
'''Calculating mean vote average and 90th percentile of vote count'''
C = movies_df["vote_average"].mean()
m = movies_df["vote_count"].quantile(0.9)

print("C: ", C)
print("m: ", m)

'''Filtering movies based on the calculated threshold'''
new_movies_df = movies_df.copy().loc[movies_df["vote_count"] >= m]
print(new_movies_df.shape)

'''Defining a weighted rating function'''
def weighted_rating(x, C=C, m=m):
    v = x["vote_count"]
    R = x["vote_average"]

    return (v / (v + m) * R) + (m / (v + m) * C)

'''Applying the weighted rating function to the filtered movies'''
new_movies_df["score"] = new_movies_df.apply(weighted_rating, axis=1)
new_movies_df = new_movies_df.sort_values('score', ascending=False)

'''Displaying the top 10 movies based on demographic filtering'''
new_movies_df[["title", "vote_count", "vote_average", "score"]].head(10)


# Plot top 10 movies
'''Plotting the top 10 movies based on popularity'''
def plot():
    popularity = movies_df.sort_values("popularity", ascending=False)
    plt.figure(figsize=(12, 6))
    plt.barh(popularity["title"].head(10), popularity["popularity"].head(10), align="center", color="skyblue")
    plt.gca().invert_yaxis()
    plt.title("Top 10 movies")
    plt.xlabel("Popularity")
    plt.show()


plot()


# Content based Filtering
'''Printing the overview of the first 5 movies'''
print(movies_df["overview"].head(5))

'''Using TfidfVectorizer to convert movie overviews into a matrix of TF-IDF features'''
tfidf = TfidfVectorizer(stop_words="english")
movies_df["overview"] = movies_df["overview"].fillna("")

tfidf_matrix = tfidf.fit_transform(movies_df["overview"])
print(tfidf_matrix.shape)


# Compute similarity
'''Computing cosine similarity between movie overviews'''
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim.shape)

'''Creating a Series with movie titles as index and movie indices as values'''
indices = pd.Series(movies_df.index, index=movies_df["title"]).drop_duplicates()
print(indices.head())


'''Defining a function to get movie recommendations based on cosine similarity'''
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

'''Processing metadata features (cast, crew, keywords, genres)'''
features = ["cast", "crew", "keywords", "genres"]

for feature in features:
    movies_df[feature] = movies_df[feature].apply(literal_eval)

movies_df[features].head(10)

'''Extracting the director from the crew information'''
def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan

'''Extracting lists of names for cast, keywords, and genres'''
def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]

        if len(names) > 3:
            names = names[:3]

        return names

    return []

'''Applying the director and list extraction functions to relevant features'''
movies_df["director"] = movies_df["crew"].apply(get_director)

features = ["cast", "keywords", "genres"]
for feature in features:
    movies_df[feature] = movies_df[feature].apply(get_list)

'''Displaying the relevant metadata information for the first 10 movies'''
movies_df[['title', 'cast', 'director', 'keywords', 'genres']].head()


'''Cleaning data by converting strings to lowercase and removing spacesv'''
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ""


'''Applying data cleaning to metadata features'''
features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)


'''Creating a "soup" by combining relevant metadata features'''
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])


movies_df["soup"] = movies_df.apply(create_soup, axis=1)
print(movies_df["soup"].head())

'''Using CountVectorizer to convert the "soup" into a matrix of token counts'''
count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(movies_df["soup"])

print(count_matrix.shape)
'''Computing cosine similarity between movies based on token counts'''
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim2.shape)

'''Resetting the DataFrame index and creating a new indices Series'''
movies_df = movies_df.reset_index()
indices = pd.Series(movies_df.index, index=movies_df['title'])

'''Displaying content-based recommendations for specific movies using metadata'''
print("################ Content Based System - metadata #############")
print("Recommendations for The Dark Knight Rises")
print(get_recommendations("The Dark Knight Rises", cosine_sim2))
print()
print("Recommendations for Avengers")
print(get_recommendations("The Avengers", cosine_sim2))
print()
print("Recommendations for The Godfather")
print(get_recommendations("The Godfather", cosine_sim2))
print()
print("Recommendations for Inception")
print(get_recommendations("Inception", cosine_sim2))
print()
print("Recommendations for The Matrix")
print(get_recommendations("The Matrix", cosine_sim2))
print()
print("Recommendations for Interstellar")
print(get_recommendations("Interstellar", cosine_sim2))
print()
print("Recommendations for Snowpiercer")
print(get_recommendations("Snowpiercer", cosine_sim2))
print()
print("Recommendations for Return of the Jedi")
print(get_recommendations("Return of the Jedi", cosine_sim2))
print()
print("Recommendations for Django Unchained")
print(get_recommendations("Django Unchained", cosine_sim2))
print()
print("Recommendations for Shrek")
print(get_recommendations("Shrek", cosine_sim2))
print()
print("Recommendations for Pacific Rim")
print(get_recommendations("Pacific Rim", cosine_sim2))
print()
print("Recommendations for Titanic")
print(get_recommendations("Titanic", cosine_sim2))
print()
print("Recommendations for Jurassic Park")
print(get_recommendations("Jurassic Park", cosine_sim2))
print()
print("Recommendations for I, Robot")
print(get_recommendations("I, Robot", cosine_sim2))
print()
print("Recommendations for Legend")
print(get_recommendations("Legend", cosine_sim2))
print()
print("Recommendations for Raiders of the Lost Ark")
print(get_recommendations("Raiders of the Lost Ark", cosine_sim2))