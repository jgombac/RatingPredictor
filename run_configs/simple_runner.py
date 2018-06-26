import os
import errno
import simple_sentiment as ss


def run_config(data_dir, reviews_filename):
    filename = data_dir + reviews_filename
    report_dir = data_dir + "simple_models/" + reviews_filename.split("/")[0] + "/confusion_matrix.csv"
    if not os.path.isfile(report_dir):
        ss.train(filename)
        confusion_matrix = ss.predictions(filename)
        create_dir(report_dir)
        confusion_matrix.to_csv(report_dir)


def create_dir(dir):
    if not os.path.exists(os.path.dirname(dir)):
        try:
            os.makedirs(os.path.dirname(dir))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise