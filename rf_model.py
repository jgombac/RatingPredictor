import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train(file_x, file_y, params):
    logging.info("Training RandomForest")
    x = file_x
    y = file_y
    if (isinstance(file_x, str)):
        x = np.load(file_x)
    if isinstance(file_y, str):
        y = np.load(file_y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=params.get("test_size"), random_state=1)

    model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_features=params["max_features"],
        n_jobs=-1
    )

    model.fit(x_train, y_train)

    return model

def run_predictions(model, file_x, file_y):
    x = file_x
    y = file_y
    if (isinstance(file_x, str)):
        x = np.load(file_x)
    if isinstance(file_y, str):
        y = np.load(file_y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    predictions = model.predict(x_test)

    pd_ratings = pd.Series(y_test[:len(predictions)], name="Actual")
    pd_predictions = pd.Series(predictions, name="Predicted")
    confusion_matrix = pd.crosstab(pd_predictions, pd_ratings)

    print(confusion_matrix)
    return confusion_matrix