import os
import errno
import logging

# configure logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# lowercase the words and remove special characters (keep the apostrophe ')
# returns array of strings
def clean(texts):
    import re

    logging.info("Cleaning")

    lines = []
    for line in texts:
        text = [w.lower() for w in re.sub('[^A-Za-z \']+', "", line).split()]
        lines.append(" ".join(text))
    return lines


# tokenize the documents
# returns array of word arrays
def tokenize(texts):
    from nltk.tokenize import RegexpTokenizer

    logging.info("Regexp Tokenizing")

    tokenizer = RegexpTokenizer(r'\w+')

    tokenized_lines = []
    for line in texts:
        tokens = tokenizer.tokenize(line)
        stop_tokens = [x.lower() for x in tokens]
        tokenized_lines.append(stop_tokens)
    return tokenized_lines


# Tokenize words (keep apostrophes ')
# returns array of word arrays
def tweet_tokenize(texts):
    from nltk.tokenize import TweetTokenizer

    logging.info("Tweet Tokenizing")

    tokenizer = TweetTokenizer()

    tokenized_lines = []
    for l in texts:
        tokens = tokenizer.tokenize(l)
        stop_tokens = [x.lower() for x in tokens]
        tokenized_lines.append(stop_tokens)
    return tokenized_lines


# Remove stop words and words with length < 1
# returns array of word arrays
def remove_stop_words(texts):
    from stop_words import get_stop_words

    logging.info("Removing stop words")

    stop_words = get_stop_words("en")

    lines = []
    for line in texts:
        removed = [w for w in line if w not in stop_words and len(w) > 1]
        lines.append(removed)
    return lines


# Lemmatize the documents
# returns array of word arrays
def lemmatize(texts):
    from nltk.stem import WordNetLemmatizer

    # download wordnet package
    import nltk
    nltk.download("wordnet")

    logging.info("Lemmatizing")

    lmt = WordNetLemmatizer()
    lines = []
    for line in texts:
        text = [lmt.lemmatize(w) for w in line]
        lines.append(text)
    return lines


# Stem the documents
# returns array of word arrays
def stem(texts):
    from nltk.stem import PorterStemmer

    logging.info("Stemming")

    porter = PorterStemmer()
    lines = []
    for line in texts:
        text = [porter.stem(word) for word in line]
        lines.append(text)
    return lines


def save_texts(texts, file_out):
    if not os.path.exists(os.path.dirname(file_out)):
        try:
            os.makedirs(os.path.dirname(file_out))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(file_out, "w+") as f:
        for line in texts:
            f.write(" ".join(line) + "\n")


def preprocess(filename, steps):
    texts = []
    with open(filename, "r") as f:
        for line in f:
            texts.append(line)

    # Cleaning
    if "clean" in steps:
        texts = clean(texts)

    # Tokenizing
    if "regexp_tokenize" in steps:
        texts = tokenize(texts)
    elif "tweet_tokenize" in steps:
        texts = tweet_tokenize(texts)

    # Removing stop words
    if "remove_stop_words" in steps:
        texts = remove_stop_words(texts)

    # Lemmatize/Stem
    if "lemmatize" in steps:
        texts = lemmatize(texts)
    elif "stem" in steps:
        texts = stem(texts)

    return texts

