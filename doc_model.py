import os
import errno
from gensim.models import doc2vec


def train_doc2vec(model_dir, train_file, train_params):
    import logging
    import multiprocessing

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    documents = doc2vec.TaggedLineDocument(train_file)

    size = train_params["size"]
    window = train_params["window"]
    min_count = train_params["min_count"]
    workers = multiprocessing.cpu_count()
    epochs = train_params["epochs"]
    alpha = train_params["alpha"]
    min_alpha = train_params["min_alpha"]

    model = doc2vec.Doc2Vec(
        documents,
        vector_size=size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        alpha=alpha,
        min_alpha=min_alpha
    )
    if not os.path.exists(os.path.dirname(model_dir)):
        try:
            os.makedirs(os.path.dirname(model_dir))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    model.save(model_dir)


def get_model(model_dir):
    return doc2vec.Doc2Vec.load(model_dir)


def test_model(model, test_file):
    from scipy import spatial
    from numpy import mean
    documents = []
    with open(test_file, "r") as f:
        for line in f:
            documents.append(line.split())
    distances = []
    for i, doc in enumerate(documents):
        trained_vec = model.docvecs[i]
        infered = model.infer_vector(doc)
        dist = spatial.distance.cosine(trained_vec, infered)
        distances.append(dist)
    print(min(distances), mean(distances), max(distances))