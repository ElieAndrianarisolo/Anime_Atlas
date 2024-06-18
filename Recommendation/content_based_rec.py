import joblib
import numpy as np
import pandas as pd
import math
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from numpy import nan
import re
from numpy import vstack
from sympy.physics.quantum.identitysearch import scipy
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import words



def extract_year(date_str):
    date_str = str(date_str)
    if 'Unknown' in date_str:
        return nan  
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', date_str)
    if len(years) == 2:  
        return (int(years[0]) + int(years[1])) // 2  
    elif years:
        return int(years[0])  
    else:
        return nan  # In case of a parsing error


def preprocess(anime_list): 
    anime_list = anime_list.copy()

    ## Dropping columns
    columns_to_keep = ['anime_id', 'Name', 'Genres', 'Synopsis', 'Episodes', 'Aired', 'Studios', 'Duration', 'Rating', 'Type', 'Source']
    anime_list = anime_list[columns_to_keep]

    ## Genres
    all_genres = set()
    for genres in anime_list['Genres']:
        all_genres.update(genres.split(', '))
    for genre in all_genres:
        anime_list["Genre " +genre] = anime_list['Genres'].apply(lambda x: 1 if genre in x.split(', ') else 0)
    anime_list.drop(columns=['Genres'], inplace=True)

    ## Episodes and Duration
    anime_list['Episodes'] = pd.to_numeric(anime_list['Episodes'], errors='coerce').fillna(0) #0 if UNKNOWN episodes
    hours = anime_list['Duration'].str.extract(r'(\d+) hr', expand=False).astype(float)
    minutes = anime_list['Duration'].str.extract(r'(\d+) min', expand=False).astype(float)
    hours.fillna(0, inplace=True)
    minutes.fillna(0, inplace=True)
    anime_list['Duration'] = hours * 60 + minutes #0 if UNKNOWN duration
    anime_list['Total_Duration'] = anime_list['Duration'] * anime_list['Episodes']

    ## Aired
    anime_list['Aired'] = anime_list['Aired'].apply(extract_year)
    middle_year = anime_list['Aired'].median()
    anime_list['Aired'] = anime_list['Aired'].fillna(middle_year)
    anime_list['Aired'] = anime_list['Aired'].astype(int)

    ## Rating
    all_ratings = set()
    for rating in anime_list['Rating']:
        all_ratings.update(rating.split(', '))
    for rating in all_ratings:
        anime_list["Rating " + rating] = anime_list['Rating'].apply(lambda x: 1 if rating in x.split(', ') else 0)
    anime_list.drop(columns=['Rating'], inplace=True)

    ## Type
    all_types = set()
    for type in anime_list['Type']:
        all_types.update(type.split(', '))
    for type in all_types:
        anime_list["Type " + type] = anime_list['Type'].apply(lambda x: 1 if type in x.split(', ') else 0)
    anime_list.drop(columns=['Type'], inplace=True)

    ## Source
    all_sources = set()
    for source in anime_list['Source']:
        all_sources.update(source.split(', '))
    for source in all_sources:
        anime_list["Source " + source] = anime_list['Source'].apply(lambda x: 1 if source in x.split(', ') else 0)
    anime_list.drop(columns=['Source'], inplace=True)

    #Synopsis
    anime_list['Synopsis'] = anime_list['Synopsis'].str.replace(r'[^\w\s]+', '')
    anime_list['Synopsis'] = anime_list['Synopsis'].str.replace('No description available for this anime.', '')
    anime_list['Synopsis'] = anime_list['Synopsis'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)).lower())
    stop_words = set(stopwords.words('english'))
    anime_list['Synopsis'] = anime_list['Synopsis'].apply(lambda x  : ' '.join([word for word in x.split() if word not in stop_words]))
    lemmatizer = WordNetLemmatizer()
    anime_list['Synopsis'] = anime_list['Synopsis'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    return anime_list

def adjust_dispersion(df, factor=0.25):
    ## Update df, which have values between 0 and 1, to adjust dispersion relatively to 0.5 to a fixed factor, while keeping the values between 0 and 1

    # Calculate the current mean absolute deviation from 0.5
    current_mad = np.abs(df['similarity'] - 0.5).mean()
    
    # Scale the values to achieve the desired dispersion relative to 0.5
    scaled_values = df['similarity'] + (0.5 - df['similarity']) * (factor / current_mad)
    
    # Ensure values are between 0 and 1
    scaled_values = np.clip(scaled_values, 0, 1)
    
    df['similarity'] = scaled_values
    
    return df


## Synopsis
def recommendation_synopsis_based(fav_anime_list, anime_list, cosine_sim):
    fav_anime_ids = [anime_id for anime_id, _ in fav_anime_list]
    anime_ids = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_ids), 'anime_id'].values
    
    if not fav_anime_list:
        return pd.DataFrame({'anime_id': anime_ids, 'similarity': 0})

    # Initialize a list to collect all similarity scores
    all_sim_scores = []

    id_list = anime_list[anime_list['anime_id'].isin(fav_anime_ids)].index

    # Compute average similarity scores from all provided indices
    for anime_id, rating in fav_anime_list:
        idx = anime_list[anime_list['anime_id'] == anime_id].index[0]  # Get index of anime in anime_list
        sim_scores = [(idx, score * (rating/10)) for idx, score in enumerate(cosine_sim[idx])]
        all_sim_scores.append(sim_scores)

    # Calculate the mean of the similarity scores across all provided indices
    mean_sim_scores = np.mean(np.array([[score for _, score in item] for item in all_sim_scores]), axis=0)

    # Create a list of tuples (index, mean score)
    mean_sim_scores = list(enumerate(mean_sim_scores))

    # Exclude the indices that were part of the input to avoid self-recommendation
    filtered_scores = [(idx, score) for idx, score in mean_sim_scores if idx not in id_list][:]

    # Get the anime indices
    anime_indices = [i[0] for i in filtered_scores]

    # Map indices to anime_id
    anime_ids = anime_list.iloc[anime_indices]['anime_id'].values
    
    # Calculate the normalized scores (L2 normalization)
    scores = np.array([i[1] for i in filtered_scores])
    norm_scores = scores / np.linalg.norm(scores)

    return pd.DataFrame({'anime_id': anime_ids, 'similarity': norm_scores})


## Date
def recommendation_date_based(fav_anime_list, anime_list):
    fav_anime_ids = [anime_id for anime_id, _ in fav_anime_list]
    anime_ids = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_ids), 'anime_id'].values
    similarities = []

    fav_date =  anime_list.loc[anime_list['anime_id'].isin(fav_anime_ids)  , 'Aired' ].mean()
    dates  = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_ids)  , 'Aired']

    for date in dates :
        similarities.append(abs(date -  fav_date))
    similarities_array = np.array(similarities).reshape(-1, 1)

    # Créer l'instance de MinMaxScaler
    scaler = MinMaxScaler()
    
    # Normaliser les similarités
    normalized_similarities = scaler.fit_transform(similarities_array)
    
    normalized_similarities  = 1 - normalized_similarities

    # Redimensionner pour revenir à une liste simple si nécessaire
    normalized_similarities = normalized_similarities.flatten().tolist()
    return pd.DataFrame({'anime_id': anime_ids, 'similarity': normalized_similarities})

## Genre
def recommendation_genre_based(fav_anime_list, anime_list):
    fav_anime_ids = [anime_id for anime_id, _ in fav_anime_list]
    anime_ids = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_ids), 'anime_id'].values

    if not fav_anime_list:
        return pd.DataFrame({'anime_id': anime_ids, 'similarity': 0})
    
    similarities = []

    temp_anime_list = anime_list.copy()
    genre_columns = anime_list.filter(like='Genre').columns.tolist()

    for anime_id, rating in fav_anime_list:
        temp_anime_list.loc[temp_anime_list['anime_id'] == anime_id, genre_columns] *= rating

    fav_genres = temp_anime_list.loc[temp_anime_list['anime_id'].isin(fav_anime_ids), temp_anime_list.filter(regex='^Genre').columns].sum()
    fav_genres_prop = fav_genres / fav_genres.sum()

    other_anime_genres = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_ids), anime_list.filter(regex='^Genre').columns]
    for _, row in other_anime_genres.iterrows():
        genre_similarity = sum(row[genre] * fav_genres_prop[genre] for genre in fav_genres_prop.index)
        similarities.append(genre_similarity)
       
    return pd.DataFrame({'anime_id': anime_ids, 'similarity': similarities})

## Rating
def recommendation_rating_based(fav_anime_list, anime_list):
    fav_anime_ids = [anime_id for anime_id, _ in fav_anime_list]
    anime_ids = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_ids), 'anime_id'].values

    if not fav_anime_list:
        return pd.DataFrame({'anime_id': anime_ids, 'similarity': 0})
    
    similarities = []

    temp_anime_list = anime_list.copy()
    rating_columns = anime_list.filter(like='Rating').columns.tolist()

    for anime_id, rating in fav_anime_list:
        temp_anime_list.loc[temp_anime_list['anime_id'] == anime_id, rating_columns] *= rating

    fav_ratings = temp_anime_list.loc[temp_anime_list['anime_id'].isin(fav_anime_ids), temp_anime_list.filter(regex='^Rating').columns].sum()
    fav_ratings_prop = fav_ratings / fav_ratings.sum()

    other_anime_ratings = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_ids), anime_list.filter(regex='^Rating').columns]
    for _, row in other_anime_ratings.iterrows():
        rating_similarity = sum(row[rate] * fav_ratings_prop[rate] for rate in fav_ratings_prop.index)
        similarities.append(rating_similarity)
       
    return pd.DataFrame({'anime_id': anime_ids, 'similarity': similarities})


## Type
def recommendation_type_based(fav_anime_list, anime_list):
    fav_anime_ids = [anime_id for anime_id, _ in fav_anime_list]
    anime_ids = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_ids), 'anime_id'].values

    if not fav_anime_list:
        return pd.DataFrame({'anime_id': anime_ids, 'similarity': 0})
    
    similarities = []

    temp_anime_list = anime_list.copy()
    type_columns = anime_list.filter(like='Type').columns.tolist()

    for anime_id, rating in fav_anime_list:
        temp_anime_list.loc[temp_anime_list['anime_id'] == anime_id, type_columns] *= rating

    fav_types = temp_anime_list.loc[temp_anime_list['anime_id'].isin(fav_anime_ids), temp_anime_list.filter(regex='^Type').columns].sum()
    fav_types_prop = fav_types / fav_types.sum()

    other_anime_types = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_ids), anime_list.filter(regex='^Type').columns]
    for _, row in other_anime_types.iterrows():
        type_similarity = sum(row[type] * fav_types_prop[type] for type in fav_types_prop.index)
        similarities.append(type_similarity)
       
    return pd.DataFrame({'anime_id': anime_ids, 'similarity': similarities})

## Source
def recommendation_source_based(fav_anime_list, anime_list):
    fav_anime_ids = [anime_id for anime_id, _ in fav_anime_list]
    anime_ids = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_ids), 'anime_id'].values

    if not fav_anime_list:
        return pd.DataFrame({'anime_id': anime_ids, 'similarity': 0})
    
    similarities = []

    temp_anime_list = anime_list.copy()
    source_columns = anime_list.filter(like='Source').columns.tolist()

    for anime_id, rating in fav_anime_list:
        temp_anime_list.loc[temp_anime_list['anime_id'] == anime_id, source_columns] *= rating

    fav_sources = temp_anime_list.loc[temp_anime_list['anime_id'].isin(fav_anime_ids), temp_anime_list.filter(regex='^Source').columns].sum()
    fav_sources_prop = fav_sources / fav_sources.sum()

    other_anime_sources = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_ids), anime_list.filter(regex='^Source').columns]
    for _, row in other_anime_sources.iterrows():
        source_similarity = sum(row[source] * fav_sources_prop[source] for source in fav_sources_prop.index)
        similarities.append(source_similarity)
       
    return pd.DataFrame({'anime_id': anime_ids, 'similarity': similarities})


## Duration
def recommendation_duration_based(fav_anime_list, anime_list):
    fav_anime_ids = [anime_id for anime_id, _ in fav_anime_list]
    anime_ids = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_ids), 'anime_id'].values

    if not fav_anime_list:
        return pd.DataFrame({'anime_id': anime_ids, 'similarity': 0})
    
    
    similarities = []

    avg_fav_duration = anime_list.loc[anime_list['anime_id'].isin(fav_anime_ids), 'Total_Duration'].mean()

    other_anime_durations = anime_list.loc[~anime_list['anime_id'].isin(fav_anime_ids), 'Total_Duration']

    for duration in other_anime_durations:
        if duration != 0:
            relative_difference = abs(duration - avg_fav_duration) / max(duration, avg_fav_duration)
            duration_similarity = 1 - relative_difference
        else:
            duration_similarity = 0.5 #similarity equals 0.5 if duration equals 0 (meaning UNKNOW number of episodes or UNKNOW duration)
        similarities.append(duration_similarity)

    return pd.DataFrame({'anime_id': anime_ids, 'similarity': similarities})


def preprocess_fav_anime_list(fav_anime_list, anime_list, feature):
    if feature == 'genre':
        filtered_fav_anime_list = [(anime_id, rating) for anime_id, rating in fav_anime_list if anime_list.loc[anime_list['anime_id'] == anime_id, 'Genre UNKNOWN'].values[0] == 0]
    elif feature == 'duration':
        filtered_fav_anime_list = [(anime_id, rating) for anime_id, rating in fav_anime_list if anime_list.loc[anime_list['anime_id'] == anime_id, 'Episodes'].values[0] != 0 and anime_list.loc[anime_list['anime_id'] == anime_id, 'Duration'].values[0] != 0]
    elif feature == 'type':
        filtered_fav_anime_list = [(anime_id, rating) for anime_id, rating in fav_anime_list if anime_list.loc[anime_list['anime_id'] == anime_id, 'Type UNKNOWN'].values[0] == 0]
    elif feature == 'source':
        filtered_fav_anime_list = [(anime_id, rating) for anime_id, rating in fav_anime_list if anime_list.loc[anime_list['anime_id'] == anime_id, 'Source Unknown'].values[0] == 0]
    elif feature == 'rating':
        filtered_fav_anime_list = [(anime_id, rating) for anime_id, rating in fav_anime_list if anime_list.loc[anime_list['anime_id'] == anime_id, 'Rating UNKNOWN'].values[0] == 0]
    elif feature == 'synopsis':
        filtered_fav_anime_list = [(anime_id, rating) for anime_id, rating in fav_anime_list if anime_list.loc[anime_list['anime_id'] == anime_id, 'Synopsis'].values[0] != '']
    else:
        filtered_fav_anime_list = fav_anime_list
    return filtered_fav_anime_list


def get_recommandation_content_tab(fav_anime_list):


    anime_list = pd.read_parquet('static/parquet/preprocessed_anime_list.parquet')

    #tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = joblib.load('Recommendation/sparse_matrix.pkl')
    #tfidf_matrix = tfidf.fit_transform(anime_list['Synopsis'])
    cosine_synopsis = linear_kernel(tfidf_matrix, tfidf_matrix)
    #cosine_synopsis = np.load('Recommandation/cosine_synopsis.npy')
    #np.save('cosine_synopsis.npy', cosine_synopsis)

    genre_cosine_similarities_tab = recommendation_genre_based(preprocess_fav_anime_list(fav_anime_list, anime_list, 'genre'), anime_list)
    genre_cosine_similarities_tab = adjust_dispersion(genre_cosine_similarities_tab)

    duration_cosine_similarities_tab = recommendation_duration_based(preprocess_fav_anime_list(fav_anime_list, anime_list, 'duration'), anime_list)
    duration_cosine_similarities_tab = adjust_dispersion(duration_cosine_similarities_tab)

    synopsis_cosine_similarities_tab = recommendation_synopsis_based(preprocess_fav_anime_list(fav_anime_list, anime_list, 'synopsis'), anime_list, cosine_synopsis)
    synopsis_cosine_similarities_tab = adjust_dispersion(synopsis_cosine_similarities_tab)

    rating_cosine_similarities_tab = recommendation_source_based(preprocess_fav_anime_list(fav_anime_list, anime_list, 'rating'), anime_list)
    rating_cosine_similarities_tab = adjust_dispersion(rating_cosine_similarities_tab)

    type_cosine_similarities_tab = recommendation_type_based(preprocess_fav_anime_list(fav_anime_list, anime_list, 'type'), anime_list)
    type_cosine_similarities_tab = adjust_dispersion(type_cosine_similarities_tab)

    source_cosine_similarities_tab = recommendation_source_based(preprocess_fav_anime_list(fav_anime_list, anime_list, 'source'), anime_list)
    source_cosine_similarities_tab = adjust_dispersion(source_cosine_similarities_tab)

    date_cosine_similarities_tab = recommendation_date_based(preprocess_fav_anime_list(fav_anime_list, anime_list, 'date'), anime_list)
    date_cosine_similarities_tab = adjust_dispersion(date_cosine_similarities_tab)

    combined_tab = pd.merge(genre_cosine_similarities_tab, duration_cosine_similarities_tab, on='anime_id', suffixes=('_genre', '_duration'))
    combined_tab = pd.merge(combined_tab, synopsis_cosine_similarities_tab, on='anime_id', suffixes=('_', '_synopsis'))
    combined_tab = pd.merge(combined_tab, type_cosine_similarities_tab, on='anime_id', suffixes=('', '_type'))
    combined_tab = pd.merge(combined_tab, source_cosine_similarities_tab, on='anime_id', suffixes=('', '_source'))
    combined_tab = pd.merge(combined_tab, rating_cosine_similarities_tab, on='anime_id', suffixes=('', '_rating'))
    combined_tab = pd.merge(combined_tab, date_cosine_similarities_tab, on='anime_id', suffixes=('', '_date'))

    # Calculate total similarity
    combined_tab['total_similarity'] = (
        0.15 * combined_tab['similarity_genre'] +
        0.01 * combined_tab['similarity_duration'] +
        0.80 * combined_tab['similarity'] + #synopsis
        0.01 * combined_tab['similarity_type'] +
        0.01 * combined_tab['similarity_source'] + 
        0.01 * combined_tab['similarity_rating'] +
        0.01 * combined_tab['similarity_date']
    )

    combined_tab = combined_tab[['anime_id', 'total_similarity']]
    combined_tab.rename(columns={'total_similarity': 'recommend_score'}, inplace=True)
    combined_tab = combined_tab.sort_values('recommend_score', ascending=False)
    return combined_tab