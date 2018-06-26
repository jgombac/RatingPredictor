import glob
import result_processor as rp


def run_configs(data_dir, reviews_filename):
    simple_dir = data_dir + "simple_models/" + reviews_filename.split("/")[0] + "/confusion_matrix.csv"
    tf_dirs = glob.glob(data_dir + "tf_models/" + reviews_filename + "*/confusion_matrix.csv")

    rp.get_metrics(simple_dir).to_csv(simple_dir.replace("confusion_matrix", "metrics_report"))

    for tf_dir in tf_dirs:
        rp.get_metrics(tf_dir).to_csv(tf_dir.replace("confusion_matrix", "metrics_report"))
