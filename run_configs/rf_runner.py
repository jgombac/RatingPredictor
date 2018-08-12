import os
import glob
import rf_model
import doc_model
import numpy as np
import errno
# import pickle
from sklearn.externals import joblib
# clf = joblib.load('filename.pk1')


def run_configs(data_dir, reviews_filename):
    doc_dir = data_dir + "doc_models/" + reviews_filename

    doc_models = glob.glob(doc_dir + "*.d2v")

    model_params = {
        "rf1": {
            "n_estimators": 100,
            "max_features": 1
        },
        "rf2": {
            "n_estimators": 100,
            "max_features": 2
        },
        "rf3": {
            "n_estimators": 100,
            "max_features": 5
        },
        "rf4": {
            "n_estimators": 100,
            "max_features": 10
        },
        "rf5": {
            "n_estimators": 100,
            "max_features": 20
        },
        "rf6": {
            "n_estimators": 100,
            "max_features": 50
        },
        "rf7": {
            "n_estimators": 100,
            "max_features": 100
        },
        "rf8": {
            "n_estimators": 100,
            "max_features": None # n_feature
        },
        "rf9": {
            "n_estimators": 100,
            "max_features": "auto" # sqrt(n_features)
        },
    }

    train_y = np.load(data_dir + reviews_filename + "_ratings.npy")
    for docmodel in doc_models:
        train_x = None

        for rfmodel in model_params:
            rf_model_name = docmodel.replace(".d2v", "_classifier_" + rfmodel + "/").replace("doc_models", "rf_models")
            create_dir(rf_model_name)

            if not os.path.isfile(rf_model_name + "model.sav.compressed"):
                if train_x is None:
                    train_x = doc_model.get_model(docmodel).docvecs.doctag_syn0
                #model_params[rfmodel]["n_inputs"] = len(train_x[0])
                model = rf_model.train(train_x, train_y, model_params[rfmodel])
                confusion_matrix = rf_model.run_predictions(model, train_x, train_y)
                joblib.dump(model, rf_model_name + "model.sav.compressed", compress=True)
                # pickle.dump(model, open(rf_model_name + "model.sav", "wb"))
                confusion_matrix.to_csv(rf_model_name + "confusion_matrix.csv")


def create_dir(dir):
    if not os.path.exists(os.path.dirname(dir)):
        try:
            os.makedirs(os.path.dirname(dir))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
