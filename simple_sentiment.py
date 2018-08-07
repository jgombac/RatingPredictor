import senticnet5 as sent_dict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# returns numpy array
def get_ratings(ratings_filename):
    return np.load(ratings_filename)


# returns array of document arrays with words
def get_reviews(reviews_filename):
    reviews = []
    with open(reviews_filename, "r") as f:
        for line in f:
            reviews.append(line.split())
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
def document_polarity(doc):
    polarity_sum = 0.0
    num_words_accounted = 0
    for i, (word1, word2) in enumerate(zip(doc)):
        current_polarity = None
        last_polarity = None

        phrase_polarity = word_polarity(word1 + "_" + word2)
        if phrase_polarity is not None:
            current_polarity = phrase_polarity
        else:
            current_polarity = word_polarity(word1)

            # if last word is not used in a phrase, try to get its polarity
            if i == len(doc) - 1:
                last_polarity = word_polarity(word2)

        if current_polarity is not None:
            polarity_sum += current_polarity
            num_words_accounted += 1
            
            # account polarity of the last word
            if last_polarity is not None:
                polarity_sum += last_polarity
                num_words_accounted += 1
    if num_words_accounted > 0:
        return polarity_sum / num_words_accounted
    return None


# calculates polarities for given txt file with documents
# saves dictionary with average document polarity at given rating and number of rating occurrences
def train(filename):
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
