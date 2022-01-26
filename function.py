import requests

import numpy as np
import pandas as pd
import ast

import nltk

from  nltk.stem.porter import PorterStemmer


from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

cv = CountVectorizer(max_features=5000,stop_words='english')

ps = PorterStemmer()


# def getMovie(genre):
#     api = "https://run.mocky.io/v3/2e0ef401-7da0-42e7-acee-6fb958e9b9e0" 
#     moviesData = requests.get(api).json()
#     genre = genre.capitalize()
#     print(genre) 
#     stringOfMovies=""
#     for movie in moviesData:
#         if (genre in movie['genres']):
#             stringOfMovies = stringOfMovies  + "------" + movie['title']  

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L=[]
    counter = 0 
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L

def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i["job"]=="Director":
            L.append(i['name'])
            break
    return L

def stem (text):
    y = []
    for i in text .split():
        y.append(ps.stem(i))
        
    return " ".join(y)

def recommend(movie, new_df, similarity):
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    result_list = []
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        result_list.append(new_df.iloc[i[0]].title)

    return result_list

def getMovie(query):

    movies = pd.read_csv("tmdb_5000_movies.csv")

    credits = pd.read_csv("tmdb_5000_credits.csv")

    movies = movies.merge(credits,on='title')
    movies=movies[["movie_id","title","overview","genres","keywords","cast","crew"]]

    movies.isnull().sum()

    movies.dropna(inplace=True)

    movies.duplicated().sum()

    movies.loc[0].genres

    movies["genres"]=movies["genres"].apply(convert)

    movies['keywords']=movies['keywords'].apply(convert)

    movies['cast']=movies['cast'].apply(convert3) 

    movies["crew"]=movies["crew"].apply(fetch_director)

    movies['overview']=movies['overview'].apply(lambda x:x.split())

    movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
    movies['keyword']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
    movies['cast']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
    movies['crew']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])

    movies['tags']=movies['overview'] + movies['genres'] + movies['keyword'] + movies['cast'] + movies['crew']

    new_df = movies[['movie_id','title','tags']]

    new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))

    new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

    new_df['tags']=new_df['tags'].apply(stem)

    vectors=cv.fit_transform(new_df['tags']).toarray()

    # cv.get_feature_names()
    ps.stem('accident')

    similarity = cosine_similarity(vectors)

    try:
        recommendation = recommend(query, new_df, similarity)
    except:
        return False

    print(type(recommendation))

    
    return recommendation



