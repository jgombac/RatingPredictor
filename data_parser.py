import gzip
import json
import pandas as pd
import numpy as np


def to_dataframe(filename):
    data = []
    i = 0
    with gzip.open(filename, "r") as f:
        for line in f:
            data.append(json.loads(line))
            # if i == 1000:
            #     break
            # i += 1
    return pd.DataFrame(data)


def get_reviews(df):
    return df["reviewText"].tolist()


# convert ratings to numpy array and make them start at 0 for classification
def get_ratings(df):
    return np.array(df["overall"].astype(np.int).tolist()) - 1


def save_ratings(ratings, file_out):
    np.save(file_out, ratings)


def save_reviews(reviews, file_out):
    with open(file_out, "w+") as f:
        for review in reviews:
            f.write(review + "\n")


def parse_data(filename):
    df = to_dataframe(filename)

    ratings = get_ratings(df)
    reviews = get_reviews(df)

    save_ratings(ratings, filename.replace(".json.gz", "_ratings.npy"))
    save_reviews(reviews, filename.replace(".json.gz", "_reviews.txt"))

