import threading
from subprocess import call
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


def board(dir):
    call(["tensorboard", "--logdir", dir, "--port", "6006"])


def board_worker(dir):
    t = threading.Thread(target=board, args=[dir])
    t.start()


def get_model(model_dir=None, params=dict()):
    n_classes = params.get("n_classes")
    feature_columns = [tf.feature_column.numeric_column("x", shape=[params.get("n_inputs")])]
    hidden_units = params.get("hidden_units")
    dropout = params.get("dropout")
    activation = params.get("activation")
    model_dir = model_dir

    optimizer = tf.train.AdamOptimizer(learning_rate=params.get("learning_rate"),
                                       beta1=params.get("beta1"),
                                       beta2=params.get("beta2"),
                                       epsilon=params.get("epsilon")
                                       )

    return tf.estimator.DNNClassifier(
        n_classes=n_classes,
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        dropout=dropout,
        activation_fn=activation,
        optimizer=optimizer,
        model_dir=model_dir
    )


def train(model, file_x, file_y, params):
    x = file_x
    y = file_y
    if (isinstance(file_x, str)):
        x = np.load(file_x)
    if isinstance(file_y, str):
        y = np.load(file_y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=params.get("test_size"), random_state=1)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_train},
        y=y_train,
        shuffle=True,
        num_epochs=None,
    )

    print("Training")
    model.train(train_input_fn, steps=params.get("epochs"))

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_test},
        y=y_test,
        shuffle=False,
        num_epochs=1,
    )

    evaluation = model.evaluate(test_input_fn)
    print("Evaluation: ", evaluation)


def run_predictions(model, file_x, file_y):
    x = file_x
    y = file_y
    if (isinstance(file_x, str)):
        x = np.load(file_x)
    if isinstance(file_y, str):
        y = np.load(file_y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": x_test},
        shuffle=False,
        num_epochs=1
    )

    print("Predicting")

    predictions = [int(prediction["classes"][0]) for prediction in list(model.predict(predict_input_fn))]
    pd_ratings = pd.Series(y_test[:len(predictions)], name="Actual")
    pd_predictions = pd.Series(predictions, name="Predicted")
    confusion_matrix = pd.crosstab(pd_predictions, pd_ratings)
    print(confusion_matrix)
    # rows = predictions
    # columns = actual
    return confusion_matrix
