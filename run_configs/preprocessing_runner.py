import os
import preprocessor

def run_configs(data_dir, reviews_filename):
    # directory where the preprocessed files will be stored
    preprocessed_dir = data_dir + "preprocessed_files/"

    # directory of raw data eg. {root}/data/electronics/reviews_Electronics_5
    filename = data_dir + reviews_filename

    # file endings
    raw = filename + ".json.gz"
    reviews = filename + "_reviews.txt"
    ratings = filename + "_ratings.npy"

    # possible preprocessing steps
    preprocess_steps = {
        "reg_lemma": ["clean", "regexp_tokenize", "remove_stop_words", "lemmatize"],
        # "reg_stem": ["clean", "regexp_tokenize", "remove_stop_words", "stem"],
        # "tw_lemma": ["clean", "tweet_tokenize", "remove_stop_words", "lemmatize"],
        # "tw_stem": ["clean", "tweet_tokenize", "remove_stop_words", "stem"],
    }

    for step in preprocess_steps:
        # generate a new filename eg. {root}/data/preprocessed_files/electronics/reviews_Electronics_5_tw_stem.txt
        preprocessed_filename = preprocessed_dir + filename.replace(data_dir, "") + "_" + step + ".txt"
        # if given file does not exist, preprocess input file with given steps and save it
        if not os.path.isfile(preprocessed_filename):
            preprocessed_texts = preprocessor.preprocess(reviews, preprocess_steps[step])
            preprocessor.save_texts(preprocessed_texts, preprocessed_filename)


