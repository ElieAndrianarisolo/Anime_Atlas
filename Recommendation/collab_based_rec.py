import numpy as np
import pandas as pd
import math


def generate_rating(liked_anime_list):
    """
        From a liked anime list, generate a defined rating and returns a DataFrame of that interaction,
        formatted similarly to the score files

        Parameters:
        - liked_book_list (list): List of favorite anime titles.
        - rating (int): The rating that you want to attribute to your favorite anime titles.

        Returns:
        DataFrame: Interaction of yourself with your favorite anime titles

    """
    interaction_list = []
    for book, rating in liked_anime_list:
        interaction_list.append([0, book, rating])
    return pd.DataFrame(interaction_list, columns=['user_id', 'anime_id', 'rating'])


def find_similar_user_interactions(data, anime_list, min_anime_similar, max_members, max_anime_reviewed):
    """
        From a data file and a list of anime
        Find users with the same type of anime taste (criteria being having at least min_anime_similar in common)

        Parameters:
        - data (DataFrame): A dictionary containing user interaction data.
        - anime_list (list): A list of favorite anime titles.
        - min_anime_similar (int): The minimum number of anime titles that must be similar
          between users for them to be considered similar.
        - max_members (int): The maximum number of members to be included.
        - max_anime_reviewed (int): The maximum number of anime a user has reviewed to consider.

        Returns:
        - list: A list of user IDs that have similar anime interactions based on the criteria.

    """
    data['same'] = data['anime_id'].isin(anime_list)
    grouped_data = data.groupby('user_id').agg(num_same=('same', 'sum'), count_review=('same', 'count')).reset_index()
    grouped_data = grouped_data[
                       (grouped_data['num_same'] >= min_anime_similar) &
                       (grouped_data['count_review'] < max_anime_reviewed)
                       ].sort_values(by='num_same', ascending=False).iloc[:max_members]
    selected_users = grouped_data['user_id'].tolist()
    return data.loc[data['user_id'].isin(selected_users)].drop('same', axis=1)


def find_total_similar_user_interactions(anime_list, coverage=1, min_anime_similar=2, max_members=100, max_anime_reviewed=500):
    """
    Finds and aggregates similar user interactions based on a set of criteria.

    This function iterates through a series of files containing user interaction data,
    applying filters to identify users with similar anime preferences. The process stops
    when the specified coverage is reached or when the maximum number of members
    or anime reviewed is exceeded.

    Parameters:
    - anime_list (list): A list of anime titles to find similar users for.
    - coverage (int, optional): The minimum number of files to process. Defaults to 1.
    - min_anime_similar (int, optional): The minimum number of anime titles that must be similar
      between users for them to be considered similar. Defaults to 2.
    - max_members (int, optional): The maximum number of members in a group to consider. Defaults to 50.
    - max_anime_reviewed (int, optional): The maximum number of anime a user has reviewed to consider. Defaults to 500.

    Returns:
    - pandas.DataFrame: A DataFrame containing aggregated user interactions that meet the criteria.

    Raises:
    - FileNotFoundError: If a specified file does not exist.
    - ValueError: If the input data is not in the expected format.
    """
    users = []
    counter = 0
    files = np.arange(5)
    np.random.shuffle(files)
    coverage_count = 0

    for i in files:
        data = pd.read_parquet(f'score/users_scores_{i}.parquet')
        temp = find_similar_user_interactions(data, anime_list, min_anime_similar, max_members - counter, max_anime_reviewed)
        users.append(temp)
        counter += len(temp)
        coverage_count += 1

        if i >= coverage:
            break

        if counter >= max_members:
            break

    return pd.concat(users, axis=0)


def create_recommendation_matrix(liked_anime_list, anime_percentage=0.6):
    """
        Creates a recommendation matrix based on user's liked anime list.

        This function generates a recommendation matrix by finding similar user interactions
        based on the user's liked anime list. It calculates the minimum number of anime
        titles that must be similar between users for them to be considered similar, based on
        a specified percentage of the total number of liked anime titles. The recommendation
        matrix is then constructed using the user's ratings and the ratings of similar users.

        Parameters:
        - liked_anime_list (list): A list of anime titles that the user has liked.
        - anime_percentage (float, optional): The percentage of liked anime titles that must be similar
          between users for them to be considered similar. Defaults to 0.6.

        Returns:
        - tuple: A tuple containing a Compressed Sparse Row (CSR) matrix representing the recommendation matrix
          and a pandas DataFrame containing the interactions used to create the matrix.

        Raises:
        - ValueError: If the input data is not in the expected format.
        - FileNotFoundError: If a specified file does not exist.
        """
    liked_anime_ids = [anime_id for anime_id, _ in liked_anime_list]
    number_anime = len(liked_anime_ids)
    min_anime_similar = math.ceil(anime_percentage * number_anime)
    similar_interactions = find_total_similar_user_interactions(liked_anime_ids, coverage=5,
                                                                min_anime_similar=min_anime_similar,
                                                                max_anime_reviewed=500)
    my_ratings = generate_rating(liked_anime_list)
    interactions = pd.concat([my_ratings, similar_interactions], axis=0)
    interactions["user_index"] = interactions["user_id"].astype("category").cat.codes
    interactions["anime_index"] = interactions["anime_id"].astype("category").cat.codes

    from scipy.sparse import coo_matrix
    rating_matrix_coo = coo_matrix((interactions['rating'], (interactions['user_index'], interactions['anime_index'])))

    return rating_matrix_coo.tocsr(), interactions


def create_anime_recommendation_df(rating_matrix, interactions):
    """
    Generates a DataFrame of anime recommendations based on cosine similarity.

    This function calculates the cosine similarity between the user's ratings and the ratings of other users. It then selects the top 15 users with the highest similarity scores, excluding the user themselves. The function aggregates the ratings of anime watched by these similar users, providing a recommendation of anime based on the mean rating and the number of times each anime was rated.

    Parameters:
    - rating_matrix (scipy.sparse.csr_matrix): A Compressed Sparse Row matrix representing user-anime interactions.
    - interactions (pandas.DataFrame): A DataFrame containing user-anime interactions, including user IDs, anime IDs, and ratings.

    Returns:
    - pandas.DataFrame: A DataFrame containing aggregated ratings for anime, including the count and mean rating for each anime.

    Raises:
    - ValueError: If the input data is not in the expected format.
    """
    my_index = interactions.loc[interactions.user_id == 0]['user_index'][0]

    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity(rating_matrix[my_index, :], rating_matrix).flatten()
    min_rec = min(similarity.shape[0], 15)
    similar_user_indices = np.argpartition(similarity, -min_rec)[-min_rec:]

    similar_users = interactions.loc[interactions['user_index'].isin(similar_user_indices)].copy()
    similar_users = similar_users.loc[similar_users['user_id'] != 0].reset_index().drop('index', axis=1)

    anime_recs = similar_users.groupby('anime_id').rating.agg(['count', 'mean'])
    return anime_recs


def recommend_collab_anime(anime_recs, liked_anime_list, min_count=2, min_mean=7):
    """
    Generates a ranked list of anime recommendations based on aggregated ratings.

    This function filters out anime that have already been watched by the user and applies
    additional criteria to select the most promising recommendations. It calculates an
    adjusted count and a recommendation score for each anime, which are used to rank the
    recommendations.

    Parameters:
    - anime_recs (pandas.DataFrame): A DataFrame containing aggregated ratings for anime,
      including the count and mean rating for each anime.
    - min_count (int, optional): The minimum number of ratings required for an anime to be
      considered for recommendation. Defaults to 2.
    - min_mean (float, optional): The minimum mean rating required for an anime to be
      considered for recommendation. Defaults to 7.

    Returns:
    - pandas.DataFrame: A DataFrame containing ranked anime recommendations, including
      the anime ID, title, count, mean rating, and recommendation score.

    Raises:
    - FileNotFoundError: If the anime list file does not exist.
    - ValueError: If the input data is not in the expected format.
    """
    anime_list = pd.read_parquet('static/parquet/anime.parquet')
    merged = pd.merge(anime_recs, anime_list, how='inner', on='anime_id')

    liked_anime_ids = [anime_id for anime_id, _ in liked_anime_list]

    # Filter out already watched anime
    merged = merged.loc[~merged['anime_id'].isin(liked_anime_ids)]
    merged = merged.loc[(merged['count'] > min_count) & (merged['mean'] > min_mean)]

    # Rank the recommendation
    merged['adjusted_count'] = merged['count'] * merged['count'] / np.exp2(np.log(merged['Scored By']))
    merged['recommend_score'] = merged['mean'] * merged['adjusted_count']
    merged.sort_values("recommend_score", ascending=False, inplace=True)

    return merged[['anime_id', 'Name', 'Score', 'Scored By', 'recommend_score']]

def get_recommandation_collab_tab(fav_anime_list):
    rating_matrix, interactions = create_recommendation_matrix(fav_anime_list, anime_percentage=0.8)
    anime_recs = create_anime_recommendation_df(rating_matrix, interactions)
    animeList = recommend_collab_anime(anime_recs, fav_anime_list) 
    animeList = animeList[['anime_id', 'recommend_score']]
    return animeList
