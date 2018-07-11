import os
import glob
import tf_model
import doc_model
import numpy as np
import tensorflow as tf
import pandas as pd
import errno

def run_configs(data_dir, reviews_filename):
    doc_dir = data_dir + "doc_models/" + reviews_filename

    doc_models = glob.glob(doc_dir + "*.d2v")

    model_params = {
        "c1": {
            "learning_rate": 0.01,
            "beta1": 0.9,
            "beta2": 0.998,
            "epsilon": 1e-08,
            "epochs": 6000,
            "test_size": 0.2,
            "dropout": 0.25,
            "n_inputs": 224,
            "hidden_units": [112],
            "n_classes": 5,
            "activation": tf.nn.relu
        },
        # "c2": {
        #     "learning_rate": 0.01,
        #     "beta1": 0.9,
        #     "beta2": 0.998,
        #     "epsilon": 1e-08,
        #     "epochs": 2000,
        #     "test_size": 0.2,
        #     "dropout": 0.3,
        #     "n_inputs": 224,
        #     "hidden_units": [224, 112],
        #     "n_classes": 5,
        #     "activation": tf.nn.relu
        # },
    }

    train_y = np.load(data_dir + reviews_filename + "_ratings.npy")
    for docmodel in doc_models:
        train_x = None

        for tfmodel in model_params:
            tf_model_name = docmodel.replace(".d2v", "_classifier_" + tfmodel).replace("doc_models", "tf_models")
            create_dir(tf_model_name)

            if not os.path.isdir(tf_model_name):
                if train_x is None:
                    train_x = doc_model.get_model(docmodel).docvecs.doctag_syn0
                model_params[tfmodel]["n_inputs"] = len(train_x[0])
                model = tf_model.get_model(tf_model_name, model_params[tfmodel])
                tf_model.train(model, train_x, train_y, model_params[tfmodel])
                confusion_matrix = tf_model.run_predictions(model, train_x, train_y)
                confusion_matrix.to_csv(tf_model_name + "/confusion_matrix.csv")


def create_dir(dir):
    if not os.path.exists(os.path.dirname(dir)):
        try:
            os.makedirs(os.path.dirname(dir))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise