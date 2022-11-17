import argparse
import json
import numpy as np


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--user1', dest='user1', required=True,
                        help='First user')
    # parser.add_argument('--user2', dest='user2', required=True,
    #         help='Second user')
    parser.add_argument("--score-type", dest="score_type", required=True,
            choices=['Euclidean', 'Pearson'], help='Similarity metric to be used')
    return parser


# Compute the Euclidean distance score between user1 and user2
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    # if user2 not in dataset:
    #     raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    common_movies = {}

    for person in dataset:
        print(person)
        for item in dataset[person]:
            if item in dataset[user1]:
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


# Compute the Pearson correlation score between user1 and user2
def pearson_score(dataset, user1):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')
    #
    # if user2 not in dataset:
    #     raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Movies rated by both user1 and user2
    temp_movies = {}
    common_movies = {}
    num_ratings = 0
    best_person = ''
    #
    # for person in dataset:
    #     if person == user1:
    #         continue
    #     for item in dataset[user1]:
    #         if item in dataset[person]:
    #             temp_movies[item] = 1
    #     if len(common_temp_moviesmovies) > num_ratings:
    #         num_ratings = len(common_movies)
    #         common_movies = {}
    #         best_person = person

    # num_ratings = len(common_movies)

    # If there are no common movies between user1 and user2, then the score is 0
    if num_ratings == 0:
        return 0

    # Calculate the sum of ratings of all the common movies
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    best_person_sum = np.sum([dataset[best_person][item] for item in common_movies])

    # Calculate the sum of squares of ratings of all the common movies
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    best_person_squared_sum = np.sum([np.square(dataset[best_person][item]) for item in common_movies])

    # Calculate the sum of products of the ratings of the common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[best_person][item] for item in common_movies])

    # Calculate the Pearson correlation score
    Sxy = sum_of_products - (user1_sum * best_person_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = best_person_squared_sum - np.square(best_person_sum) / num_ratings

    if Sxx * Syy == 0:
        return 0

    print(best_person)
    return Sxy / np.sqrt(Sxx * Syy)


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user1 = args.user1
    score_type = args.score_type

    ratings_file = 'rates.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    all_scores = {}
    for user in data:
        if user == user1:
            continue
        all_scores[user] = euclidean_score(data, user1, user)

    print(max(all_scores.values()))



    if score_type == 'Euclidean':
        print("\nEuclidean score:")
        print(euclidean_score(data, user1, user1))
    else:
        print("\nPearson score:")
        print(pearson_score(data, user1))

