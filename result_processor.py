import pandas as pd
import numpy as np


# returns dataframe with accuracy, precision and recall for each rating given confusion matrix file
def get_metrics(cm_filename):
    # read csv -> transform to numpy matrix and remove first column (indexes/classes)
    # rows = predictions
    # columns = actual
    csv = pd.read_csv(cm_filename)

    # combine original matrix with an empty one in case any of the columns are missing
    empty_df = pd.DataFrame(
        [[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0], [3, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0]],
        columns=["Predicted", "0", "1", "2", "3", "4"]
    )
    csv = pd.concat([csv, empty_df], axis=0).groupby("Predicted").sum().reset_index()

    cross_matrix = csv.as_matrix()[:, 1:]

    # true positives
    tp = np.diag(cross_matrix)
    # false positives
    fp = np.sum(cross_matrix, axis=1) - tp
    # false negatives
    fn = np.sum(cross_matrix, axis=0) - tp

    num_classes = len(tp)
    # true negatives
    tn = []
    for i in range(num_classes):
        temp = np.delete(cross_matrix, i, 0)
        temp = np.delete(temp, i, 1)
        tn.append(sum(sum(temp)))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = np.nan_to_num(tp / (tp + fp))  # or positive predictive value
    recall = np.nan_to_num(tp / (tp + fn))     # or sensitivity

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall
    }

    return pd.DataFrame(metrics)
