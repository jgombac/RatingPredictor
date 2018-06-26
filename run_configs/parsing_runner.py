import os
import data_parser


def run_configs(data_dir, reviews_filename):
    # directory of raw data eg. {root}/data/preprocessed_files/electronics/reviews_Electronics_5
    filename = data_dir + reviews_filename

    # file endings
    raw = filename + ".json.gz"
    reviews = filename + "_reviews.txt"
    ratings = filename + "_ratings.npy"

    # if review or rating file doesnt exist, parse data and create them
    if not os.path.isfile(reviews) or not os.path.isfile(ratings):
        data_parser.parse_data(raw)