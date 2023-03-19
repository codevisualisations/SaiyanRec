from django.urls import reverse
from django.http import HttpResponseRedirect
from django.views.generic import TemplateView

from django.shortcuts import render

from tensorflow import keras
from keras.models import load_model

from keras import backend as K
from collections import defaultdict

from django.contrib.auth import get_user_model
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required

from rake_nltk import Rake
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


import numpy as np
import pandas as pd

User = get_user_model()


#the test page, and thanks page, setup in settings.py is visualised as a class here
#these are then connected in urls.py
class TestPage(TemplateView):
    template_name = 'test.html'

class ThanksPage(TemplateView):
    template_name = 'thanks.html'

class HomePage(TemplateView):
    template_name = 'index.html'

    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return HttpResponseRedirect(reverse("test"))
        return super().get(request, *args, **kwargs)





#source: https://stackoverflow.com/questions/43855162/rmse-rmsle-loss-function-in-keras
def root_mean_squared_error(y_true, y_pred):
  return K.sqrt(K.mean(K.square(y_pred - y_true)))

model = load_model('my_model.h5', custom_objects={"root_mean_squared_error": root_mean_squared_error})
rating_complete= pd.read_csv('rating_complete.csv')

ratings_number = rating_complete['user_id'].value_counts()
rating_complete = rating_complete[rating_complete['user_id'].isin(ratings_number[ratings_number >= 50].index)].copy()

#source: https://keras.io/examples/structured_data/collaborative_filtering_movielens/
user_ids = rating_complete["user_id"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded2user = {i: x for i, x in enumerate(user_ids)}
rating_complete["user"] = rating_complete["user_id"].map(user2user_encoded)

#source: https://keras.io/examples/structured_data/collaborative_filtering_movielens/
anime_ids = rating_complete["anime_id"].unique().tolist()
anime2anime_encoded = {x: i for i, x in enumerate(anime_ids)}
anime_encoded2anime = {i: x for i, x in enumerate(anime_ids)}
rating_complete["anime"] = rating_complete["anime_id"].map(anime2anime_encoded)

#source: https://www.kaggle.com/willkoehrsen/neural-network-embedding-recommendation-system
def extract_weights(name, model):
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]
    weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
    return weights
anime_weights = extract_weights('anime_embedding', model)
user_weights = extract_weights('user_embedding', model)






df = pd.read_csv('anime.csv')
df = df.replace("Unknown", np.nan)

#source: https://www.kaggle.com/chaitanya99/recommendation-system-cf-anime
def getAnimeName(anime_id):
    try:
        name = df[df.anime_id == anime_id].eng_version.values[0]
        if name is np.nan:
            name = df[df.anime_id == anime_id].Name.values[0]
    except:
        return("Couldn't find Anime Name")

    return name

df['anime_id'] = df['MAL_ID']
df["eng_version"] = df['English name']
df['eng_version'] = df.anime_id.apply(lambda x: getAnimeName(x))
#https://www.kaggle.com/shadey/anime-recommendation-system-cf/comments
df.sort_values(by=['Score'], inplace=True,
                      ascending=False, kind='quicksort',
                      na_position='last')

df = df[["anime_id", "eng_version",
         "Score", "Genders", "Episodes",
         "Type", "Premiered", "Members"]]

#source: https://www.kaggle.com/chaitanya99/recommendation-system-cf-anime
def getAnimeFrame(anime):
    if isinstance(anime, int):
        return df[df.anime_id == anime]
    if isinstance(anime, str):
        return df[df.eng_version == anime]

designatedcolumns = ["MAL_ID", "Name", "Genders", "sypnopsis"]
SDF = pd.read_csv('anime_with_synopsis.csv', usecols=designatedcolumns)

#source: https://www.kaggle.com/chaitanya99/recommendation-system-cf-anime
def getSypnopsis(anime):
    if isinstance(anime, int):
        return SDF[SDF.MAL_ID == anime].sypnopsis.values[0]
    if isinstance(anime, str):
        return SDF[SDF.Name == anime].sypnopsis.values[0]







#source: https://www.kaggle.com/chaitanya99/recommendation-system-cf-anime
#source: https://www.kaggle.com/willkoehrsen/neural-network-embedding-recommendation-system#Extract-Embeddings-and-Analyze
def AnimePredictions(name, n=10, return_dist=False, neg=False):
    try:
        index = getAnimeFrame(name).anime_id.values[0]
        encoded_index = anime2anime_encoded.get(index)
        weights = anime_weights

        dotproduct = np.dot(weights, weights[encoded_index])
        argsort = np.argsort(dotproduct)

        n = n + 1

        if neg:
            similar = argsort[:n]
        else:
            similar = argsort[-n:]

        if return_dist:
            return dotproduct, similar

        rindex = df

        SimilarityArr = []

        for close in similar:
            decoded_id = anime_encoded2anime.get(close)
            sypnopsis = getSypnopsis(decoded_id)
            anime_frame = getAnimeFrame(decoded_id)

            anime_name = anime_frame.eng_version.values[0]

            if anime_name is np.nan:
                anime_name = anime_frame.Name.values[0]

            genre = anime_frame.Genders.values[0]

            similarity = dotproduct[close]

            SimilarityArr.append({"anime_id": decoded_id, "name": anime_name,
                                  "similarity": similarity,"genre": genre,
                                  'sypnopsis': sypnopsis})


        Frame = pd.DataFrame(SimilarityArr)
        Frame = Frame.rename(columns={'name':'Anime Name:'})
        Frame = Frame.set_index('Anime Name:')
        Frame = Frame.sort_values(by="similarity", ascending=False)
        Frame = Frame.rename(columns = {'similarity':'Similarity Score (%)','sypnopsis':'Synopsis', 'genre':'Genre'},inplace=False)
        c = Frame[Frame.anime_id != index].drop(['anime_id'], axis=1)
        c['Similarity Score (%)'] = c['Similarity Score (%)'].multiply(100)

        htmlframe = c.to_html()
        return htmlframe

    except:
        return('{}, Not Found in Anime list'.format(name))

def recommend(request):
    return render(request, 'recommend.html')

def result(request):
    name = str(request.GET['Anime Name'])

    result = AnimePredictions(name, n=10, return_dist=False, neg=False)

    return render(request, 'result.html', {'result':result})






#source: https://www.kaggle.com/chaitanya99/recommendation-system-cf-anime
#source: https://www.kaggle.com/willkoehrsen/neural-network-embedding-recommendation-system
def usersimilarity(pid, n=10,return_dist=False, neg=False):
    try:
        index = pid
        encoded_index = user2user_encoded.get(index)
        weights = user_weights

        dotproduct = np.dot(weights, weights[encoded_index])
        argsort = np.argsort(dotproduct)

        n = n + 1

        if neg:
            similar = argsort[:n]
        else:
            similar = argsort[-n:]

        if return_dist:
            return dotproduct, similar

        rindex = df
        SimilarityArr = []

        for close in similar:
            similarity = dotproduct[close]

            if isinstance(pid, int):
                decoded_id = user_encoded2user.get(close)
                SimilarityArr.append({"similar_users": decoded_id,
                                      "similarity": similarity})

        Frame = pd.DataFrame(SimilarityArr)
        Frame = Frame.rename(columns = {'similar_users':'Similar Users'})
        Frame = Frame.set_index('Similar Users')
        Frame = Frame.sort_values(by="similarity", ascending=False)
        Frame = Frame.rename(columns = {'similarity':'Similarity Score (%)'})
        Frame['Similarity Score (%)'] = Frame['Similarity Score (%)'].multiply(100)
        Frame = Frame.tail(Frame.shape[0] -1)

        htmlframe2 = Frame.to_html()
        return htmlframe2
    except:
        return'{}!, Not Found in User list'.format(pid)

#source: https://www.kaggle.com/chaitanya99/recommendation-system-cf-anime
def useranimepreferences(user_id, plot=False, verbose=0):
    watched_by_user = rating_complete[rating_complete.user_id==user_id]
    percentile = np.percentile(watched_by_user.rating, 75)
    watched_by_user = watched_by_user[watched_by_user.rating >= percentile]
    bestanimes = (
        watched_by_user.sort_values(by="rating", ascending=False)#.head(10)
        .anime_id.values
    )

    concatenate = df[df["anime_id"].isin(bestanimes)]
    concatenate = concatenate[["eng_version","Score", "Genders"]]

    g = concatenate
    g = g.rename(columns={'eng_version':'Anime Name:'})
    g = g.set_index('Anime Name:')
    g = g.rename(columns = {'Genders':'Genres'},inplace=False)
    htmlframe3 = g.to_html()
    return htmlframe3


def recommend2(request):
    return render(request, 'recommend2.html')

def result2(request):
    pid = int(request.GET['pid'])

    result2 = usersimilarity(pid, n=10,return_dist=False, neg=False), useranimepreferences(pid, plot=False, verbose=0)

    return render(request, 'result2.html', {'result2':result2})




#source: https://keras.io/examples/structured_data/collaborative_filtering_movielens/
def unseen_anime(user_id, n=10):
    watched_by_user = rating_complete[rating_complete.user_id==user_id]
    not_watched = df[
        ~df["anime_id"].isin(watched_by_user.anime_id.values)
    ]

    totalunseen = list(
        set(not_watched['anime_id']).intersection(set(anime2anime_encoded.keys()))
    )

    totalunseen = [[anime2anime_encoded.get(x)] for x in totalunseen]

    idencoded = user2user_encoded.get(user_id)

    ua_array = np.hstack(
        ([[idencoded]] * len(totalunseen), totalunseen)
    )

    ua_array = [ua_array[:, 0], ua_array[:, 1]]
    ratings = model.predict(ua_array).flatten()

    top_ratings_indices = (-ratings).argsort()[:10]

    recommendations = [
        anime_encoded2anime.get(totalunseen[x][0]) for x in top_ratings_indices
    ]

    Results = []
    ids = []
    #source: https://www.kaggle.com/chaitanya99/recommendation-system-cf-anime
    #source: https://www.kaggle.com/shadey/anime-recommendation-system-cf/comments
    for index, anime_id in enumerate(totalunseen):
        rating = ratings[index]
        id_ = anime_encoded2anime.get(anime_id[0])

        if id_ in recommendations:
            ids.append(id_)
            try:
                condition = (df.anime_id == id_)
                name = df[condition]['eng_version'].values[0]
                genre = df[condition].Genders.values[0]
                score = df[condition].Score.values[0]
                sypnopsis = getSypnopsis(int(id_))
            except:
                continue

            Results.append({
                            "name": name,
                            "pred_rating": rating,
                            "genre": genre,
                            'sypnopsis': sypnopsis})

    Results = pd.DataFrame(Results)
    Results = Results.rename(columns={'name':'Anime Name:'})
    Results = Results.set_index('Anime Name:')
    Results = Results.sort_values(by='pred_rating', ascending=False)
    Results = Results.rename(columns = {'pred_rating':'Similarity Score (%)', 'sypnopsis':'Synopsis', 'genre':'Genre'},inplace=False)
    Results['Similarity Score (%)'] = Results['Similarity Score (%)'].multiply(100)
    htmlframe4 = Results.to_html()
    return htmlframe4

def my_view(request):
    username = None
    if request.user.is_authenticated:
        username = int(request.user.username)
        result3 = unseen_anime(username, n=10)
        return render(request,'result3.html', {'result3':result3})
    else:
        pass





new = pd.read_csv('combined.csv')
#source: https://www.kdnuggets.com/2019/11/content-based-recommender-using-natural-language-processing-nlp.html
new['Key_words'] = ''
rake = Rake()
for index, row in new.iterrows():
    rake.extract_keywords_from_text(row['Synopsis'])
    scores = rake.get_word_degrees()
    row['Key_words'] = list(scores.keys())

new['Genres'] = new['Genres'].map(lambda x: x.split(','))

for index, row in new.iterrows():
    row['Genres'] = [x.lower().replace(' ','') for x in row['Genres']]

new['Bag_of_words'] = ''
cols = ['Genres','Key_words']
#source: https://www.kaggle.com/sasha18/recommender-systems-based-on-content
for index, row in new.iterrows():
    words = ''
    for col in cols:
        words += ' '.join(row[col]) + ' '
    row['Bag_of_words'] = words

new = new[['Name','Synopsis','Genres','Japanese Name','Bag_of_words']]

new2 = new.copy()
new2 = new.rename(columns={'Name':0})
del new2['Bag_of_words']
new2 = new2[[0,'Japanese Name','Genres','Synopsis']]
new2['Synopsis'] = new2['Synopsis'].str.replace('No synopsis information has been added to this title. Help improve our database by adding a synopsis here .','')


tfidf = TfidfVectorizer(stop_words='english')
mymatrix = tfidf.fit_transform(new['Bag_of_words'])

cosine_sim2 = linear_kernel(mymatrix, mymatrix)
indices = pd.Series(new['Name'])

#source: https://gist.github.com/emmagrimaldi/4e33c0091d2294b04c063b552925fe5f
def rarerecommender(title, cosine_sim2 = cosine_sim2):
    rare_animes= []

    p = indices[indices == title].index[0]
    values = pd.Series(cosine_sim2[p]).sort_values(ascending = False)

    top_p = list(values.iloc[1:16].index)
    top_values = list(values[1:16])
    simscores = np.array(top_values)

    for i in top_p:
        rare_animes.append(list(new['Name'])[i])

    Frame = pd.DataFrame(rare_animes)
    Frame['Similarity Score (%)'] = simscores

    df3 = pd.merge(Frame, new2, on=0)
    df3 = df3.rename(columns={0:'Name'})
    df3 = df3.set_index('Name')
    df3['Similarity Score (%)'] = df3['Similarity Score (%)'].multiply(100)
    htmlframe5 = df3.to_html()

    return htmlframe5


def newview(request):
    name = str(request.GET['dropdown'])

    result4 = rarerecommender(name, cosine_sim2 = cosine_sim2)

    return render(request, 'result4.html', {'result4': result4})


def newview2(request):
    name2 = str(request.GET['Niche Name'])

    result5 = rarerecommender(name2, cosine_sim2 = cosine_sim2)

    return render(request, 'result5.html', {'result5':result5})
