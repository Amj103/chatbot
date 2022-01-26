from rivescript import RiveScript 
from flask import Flask , request , render_template
import requests 
import bot
from function import getMovie


import numpy as np
import pandas as pd
import ast

import nltk

from  nltk.stem.porter import PorterStemmer


from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import time

cv = CountVectorizer(max_features=5000,stop_words='english')

ps = PorterStemmer()


app = Flask(__name__)




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


## pre-loading stuff

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


@app.route("/")
def index():
    bot.bot.clear_uservars()
    return render_template("index.html")

@app.route("/get")
def chat():

    time.sleep(5)

    request_data = request.args.get('msg')
    # request_data = request_data.capitalize()
    if "recommend movies like" in request_data:

        mov_split = request_data.split("like ")[1].capitalize()
        print("mov_split: ",mov_split)
        
        try:
            movieToPrint = recommend(mov_split, new_df, similarity)
            to_print = "Recommended Movies: "
            if movieToPrint is False:
                return "Sorry! No similar movies found!"
            for movie in movieToPrint:
                to_print= to_print+" | "+ movie
            return to_print
        except:
            return "Sorry! No similar movies found!"
       
    else:
        response = bot.chat(request_data)
        return str(response)  

if __name__ == "__main__":

    app.run(debug=True)