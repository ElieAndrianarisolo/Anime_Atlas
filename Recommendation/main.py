import os
import time
from Recommendation.collab_based_rec import *
from Recommendation.content_based_rec import *

fav_anime_list = [(21, 10), (1, 5)]
# filter_name = 0
anime_list = pd.read_parquet('static/parquet/anime.parquet')


def filter_anime_name(fav_anime_list, recommended_animes):
    """
            Filters the anime_list based on their anime. If a recommended anime is contained
            in one of the series of fav_anime_list , it is removed

            Parameters:
            - fav_anime_list (Python List): a list containing the ids of the favorite animes and their rating.
            -recommended_animes (List of Anime_ids) : a list containing the recommended_animes

            Returns:
            - the List of the favorite animes filtered
            """
    fav_anime_ids = [anime_id for anime_id, _ in fav_anime_list]
    fav_anime_list_names = anime_list[anime_list['anime_id'].isin(fav_anime_ids)]['Name']
    recommended_animes_names = recommended_animes['Name']
    vectorizer = TfidfVectorizer(stop_words='english').fit(anime_list['Name'])  # Entraînement sur tous les noms d'anime
    fav_vectors = vectorizer.transform(fav_anime_list_names.values)
    rec_vectors = vectorizer.transform(recommended_animes_names.values)
    similarity_matrix = linear_kernel(rec_vectors, fav_vectors)
    threshold = 0.2
    similar_indices = (similarity_matrix > threshold).any(axis=1)
    filtered_recommendation = recommended_animes[~similar_indices]
    filtered_recommendation = filtered_recommendation[['anime_id']]
    return filtered_recommendation.head(30)['anime_id'].tolist()




def show_names(anime_ids, anime_list):
    animes = []
    for anime_id in anime_ids:
        anime_name = anime_list.loc[anime_list['anime_id'] == anime_id, 'Name'].iloc[0]
        animes.append({'anime_id': anime_id, 'Name': anime_name})

    print(pd.DataFrame(animes))


def merge_score(collab_tab, content_tab, parameter):
    final_tab = pd.merge(content_tab, collab_tab, on='anime_id', suffixes=('_content', '_collab'), how='left')
    final_tab['recommend_score_collab'] = final_tab['recommend_score_collab'].fillna(0)
    final_tab['recommend_score_content'] = (final_tab['recommend_score_content'] - final_tab[
        'recommend_score_content'].min()) / (final_tab['recommend_score_content'].max() - final_tab[
        'recommend_score_content'].min())
    final_tab['recommend_score_collab'] = (final_tab['recommend_score_collab'] - final_tab[
        'recommend_score_collab'].min()) / (final_tab['recommend_score_collab'].max() - final_tab[
        'recommend_score_collab'].min())
    final_tab['total_score'] = parameter*final_tab['recommend_score_content'] + (1-parameter)*final_tab['recommend_score_collab']
    return final_tab


def recommendation_anime(fav_anime_list, filter_name=0):
    """
            Generates a ranked list of anime recommendations based on the merge of the collaborative filtering and content Filtering .

            Parameters:
            - fav_anime_list (Python List): a list containing the ids of the favorite animes and their rating.
            -filter_name (int, optional) = if equal 1 returns the Favorite animes that does not belong to the same serie

            Returns:
            - the List of the 200 favorite animes
    """
    collab_tab = get_recommandation_collab_tab(fav_anime_list)
    content_tab = get_recommandation_content_tab(fav_anime_list)

    similarities_tab = merge_score(collab_tab, content_tab)

    sorted_df = similarities_tab.sort_values(by='total_score', ascending=False)
    top_anime_ids = sorted_df.head(200)['anime_id'].tolist()

    if filter_name == 1:
        recommended_animes = []
        for anime_id in top_anime_ids:
            anime_name = anime_list.loc[anime_list['anime_id'] == anime_id, 'Name'].iloc[0]
            recommended_animes.append({'anime_id': anime_id, 'Name': anime_name})

        anime_names = pd.DataFrame(recommended_animes)
        top_anime_ids = filter_anime_name(fav_anime_list, anime_names)

    return top_anime_ids

def filter_anime_name_based(fav_anime_list , recommended_anime_ids):
    recommended_animes = []
    for anime_id in recommended_anime_ids:
        anime_name = anime_list.loc[anime_list['anime_id'] == anime_id, 'Name'].iloc[0]
        recommended_animes.append({'anime_id': anime_id, 'Name': anime_name})
    anime_names = pd.DataFrame(recommended_animes)
    top_anime_ids = filter_anime_name(fav_anime_list, anime_names)
    return top_anime_ids

def recommandation_anime_merge(collab_tab, content_tab, parameter=0.5):
    similarities_tab = merge_score(collab_tab, content_tab, parameter)
    
    return similarities_tab


def recommandation_anime_content_based(fav_anime_list):
    """
        Generates a ranked list of anime recommendations based on the content of the favortie animes  and their ratings.

        Parameters:
        - fav_anime_list (Python List): a list containing the ids of the favorite animes and their rating.
        -filter_name (int, optional) = if equal 1 returns the Favorite animes that does not belong to the same serie

        Returns:
        - the List of the 200 favorite animes
    """
    content_tab = get_recommandation_content_tab(fav_anime_list)

    return content_tab


def recommandation_anime_collab_based(fav_anime_list):
    """
            Generates a ranked list of anime recommendations based on the ratings of the other users.

            Parameters:
            - fav_anime_list (Python List): a list containing the ids of the favorite animes and their rating.
            -filter_name (int, optional) = if equal 1 returns the Favorite animes that does not belong to the same serie

            Returns:
            - the List of the 200 favorite animes
    """
    collab_tab = get_recommandation_collab_tab(fav_anime_list)
    print(collab_tab.shape)

    return collab_tab


def sortAndFormat(tab):
    sorted_df = tab.sort_values(by='recommend_score', ascending=False)
    top_anime_ids = sorted_df.head(200)['anime_id'].tolist()
    return top_anime_ids[:200]
    
def sortMergeAndFormat(tab):
    sorted_df = tab.sort_values(by='total_score', ascending=False)
    top_anime_ids = sorted_df.head(200)['anime_id'].tolist()
    return top_anime_ids[:200]


if __name__ == "__main__":
    # anime_list = pd.read_parquet('../static/parquet/anime.parquet')
    start_time = time.time()
    #recommandation_anime_collab_based(fav_anime_list)

    # show_names(recommendation_anime(fav_anime_list, 1), anime_list)

    show_names(recommandation_anime_collab_based(fav_anime_list, 0), anime_list)

    # recommandation_anime_collab_based(fav_anime_list, 0)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    # show_names(recommandation_anime_collab_based(fav_anime_list, 0), anime_list)