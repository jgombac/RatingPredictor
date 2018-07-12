import os
import doc_model

def run_configs(data_dir, reviews_filename):
    # directory where the preprocessed files are stored
    preprocessed_dir = data_dir + "preprocessed_files/"

    # directory of raw data eg. {root}/data/preprocessed_files/electronics/reviews_Electronics_5
    filename = data_dir + reviews_filename

    preprocessed_steps = [
        "reg_lemma",
        # "reg_stem",
        # "tw_lemma",
        # "tw_stem"
    ]

    train_configs = {
        "t1": {
            "size": 224,
            "window": 4,
            "min_count": 10,
            "epochs": 40,
            "alpha": 0.05,
            "min_alpha": 1e-4
        },
        "t2": {
            "size": 264,
            "window": 4,
            "min_count": 20,
            "epochs": 100,
            "alpha": 0.05,
            "min_alpha": 1e-4
        },
        "t3": {
            "size": 264,
            "window": 2,
            "min_count": 20,
            "epochs": 60,
            "alpha": 0.05,
            "min_alpha": 1e-4
        },
    }

    # iterate over preprocessed files
    for step in preprocessed_steps:
        preprocessed_filename = preprocessed_dir + filename.replace(data_dir, "") + "_" + step + ".txt"
        for train_config in train_configs:
            # directory where the doc2vec model will be saved eg. {root}/data/doc_models/electronics/reviews_Electronics_5_tw_stem_t2.d2v
            doc_model_name = preprocessed_filename.replace(".txt", "_" + train_config + ".d2v").replace("preprocessed_files", "doc_models")
            if not os.path.isfile(doc_model_name):
                doc_model.train_doc2vec(doc_model_name, preprocessed_filename, train_configs[train_config])


