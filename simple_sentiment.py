import senticnet5 as sent_dict
import pandas as pd
import numpy as np
from itertools import islice
from sklearn.model_selection import train_test_split
import re


# returns numpy array
def get_ratings(ratings_filename):
    return np.load(ratings_filename)


# returns array of document arrays with words
def get_reviews(reviews_filename):
    reviews = []
    with open(reviews_filename, "r") as f:
        for line in f:
            reviews.append([w.lower() for w in re.sub('[^A-Za-z \']+', "", line).split()])
    return reviews


# returns word polarity: float
# if word not in dictionary return None
def word_polarity(word):
    try:
        return float(sent_dict.senticnet[word][7])
    except:
        return None


# return average polarity of a given document
# if none of the words are in dictionary return None
# accounts all single words and combinations of 2 words
def document_polarity(doc):
    polarity_sum = 0.0
    num_words_accounted = 0
    phrases = get_phrases(doc, 2)
    for phrase in phrases:
        current_polarity = word_polarity(phrase)

        if current_polarity is not None:
            polarity_sum += current_polarity
            num_words_accounted += 1

    if num_words_accounted > 0:
        return polarity_sum / num_words_accounted
    return None


# calculates polarities for given txt file with documents
# saves dictionary with average document polarity at given rating and number of rating occurrences
def train(filename):
    print("TRAINING SIMPLE SENTIMENT")
    results = {
        0.0: [0.0, 0],          # average polarity at given rating
        1.0: [0.0, 0],
        2.0: [0.0, 0],
        3.0: [0.0, 0],
        4.0: [0.0, 0],
        "Undefined": [0.0, 0]   # if polarity can't be determined use this to determine average rating for such occurrences
    }

    ratings = get_ratings(filename + "_ratings.npy")
    reviews = get_reviews(filename + "_reviews.txt")

    x_train, x_test, y_train, y_test = train_test_split(reviews, ratings, test_size=0.2, random_state=1)

    for doc, rating in zip(x_train, y_train):
        polarity = document_polarity(doc)

        if polarity is None:
            results["Undefined"][0] += rating
            results["Undefined"][1] += 1
        else:
            results[rating][0] += polarity
            results[rating][1] += 1

    for key in results:
        results[key][0] = results[key][0] / max(results[key][1], 1)

    pd.DataFrame(results).to_csv(filename + "_polarities.csv")


# gives rating prediction based on closest average document polarity
def predictions(filename):
    print("PREDICTING SIMPLE SENTIMENT")
    predictions = []
    ratings = get_ratings(filename + "_ratings.npy")
    reviews = get_reviews(filename + "_reviews.txt")
    rating_polarities = pd.read_csv(filename + "_polarities.csv")
    default_rating = float(round(rating_polarities.loc[0, "Undefined"]))
    polarities = rating_polarities[["0.0", "1.0", "2.0", "3.0", "4.0"]].iloc[0].tolist()

    x_train, x_test, y_train, y_test = train_test_split(reviews, ratings, test_size=0.2, random_state=1)

    for doc, rating in zip(x_test, y_test):
        polarity = document_polarity(doc)
        prediction = default_rating
        if polarity is not None:
            prediction = float(polarities.index(min(polarities, key=lambda x:abs(x - polarity))))
        predictions.append(prediction)
    pd_ratings = pd.Series(ratings[:len(predictions)], name="Actual")
    pd_predictions = pd.Series(predictions, name="Predicted")
    confusion_matrix = pd.crosstab(pd_predictions, pd_ratings)
    return confusion_matrix


# generates exhaustible sliding window over a sequence
# [1, 2, 3, 4], 2 => 12 23, 34, 4
# [1, 2, 3, 4], 3 => 123, 234, 34, 4
def get_windows(sequence, n):
    windows = []
    for i, x in enumerate(sequence):
        windows.append(list(islice(sequence, i, i+n)))
    return windows


# returns all combinations retaining the order
# eg. 1, 2, 3 => 1, 1_2, 1_2_3
def get_combinations(sequence):
    combinations = []
    for i, x in enumerate(sequence):
        combinations.append("_".join(sequence[:i] + [x]))
    return combinations


# returns all posible combinations with a sliding window
# eg. window_size = 2
# 1, 2, 3, 4 => 1, 1_2, 2, 2_3, 3, 3_4,
def get_phrases(doc, window_size):
    phrases = []
    for window in get_windows(doc, window_size):
        phrases += get_combinations(window)
    return phrases

