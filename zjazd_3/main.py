# Authors: Marcin Żmuda-Trzebiatowski and Jakub Cirocki
#

import argparse
import json
import numpy as np


def build_arg_parser():
    """
    Building arguments from console
    """
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--user', dest='user', required=True,
                        help='User')

    return parser


def euclidean_score(dataset, user1, user2):
    """
    Calculating euclidean score
    """

    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

        # Movies rated by both user1 and user2
    common_movies = {}

    for item in dataset[user1]:

        if item in dataset[user2]:
            common_movies[item] = 1

    # If there are no common movies between the users,
    # then the score is 0

    if len(common_movies) == 0:
        return 0

    squared_diff = []

    for item in dataset[user1]:

        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


def user_best_movies(dataset, user1, user2):
    """
    Getting the best movies from user
    """
    user1_movies = dataset[user1]
    user2_movies = dataset[user2]

    user2_movies_sorted = dict(sorted(user2_movies.items(), key=lambda item: item[1], reverse=True))

    return dict(filter(lambda item: item[1] > 5 and item[0] not in user1_movies, user2_movies_sorted.items()))


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user_input = args.user

    ratings_file = 'rates.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    all_scores = {}
    for user in data:
        if user == user_input:
            continue
        all_scores[user] = euclidean_score(data, user_input, user)

    scores_sorted_asc = dict(sorted(all_scores.items(), key=lambda item: item[1]))
    scores_sorted_desc = dict(sorted(all_scores.items(), key=lambda item: item[1], reverse=True))

    best_movies = {}
    for user in scores_sorted_desc:
        while len(best_movies) < 6:
            best_movies = user_best_movies(data, user_input, user) | best_movies

    print("Rekomendacje")
    best_movies = list(best_movies.keys())
    for movie_index in range(5):
        print(best_movies[movie_index])

    print("\n")

    worst_movies = {}
    for user in scores_sorted_asc:
        while len(worst_movies) < 6:
            worst_movies = user_best_movies(data, user_input, user) | worst_movies

    print("Nie oglądaj")
    worst_movies = list(worst_movies.keys())
    for movie_index in range(5):
        print(worst_movies[movie_index])
