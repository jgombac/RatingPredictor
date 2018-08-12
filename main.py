import os
import glob
import run_config as runner


def get_subdirectories(parent_dir):
    return [dir for dir in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, dir))]


def remove_project_dirs(dir_list):
    project_dirs = ["doc_models", "preprocessed_files", "simple_models", "tf_models", "rf_models", "books"]
    return [dir for dir in dir_list if dir not in project_dirs]


if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.realpath(__file__)) + "/data/"
    subdirectories = remove_project_dirs(get_subdirectories(data_dir))

    for subdir in subdirectories:
        raw_reviews_filenames = glob.glob(data_dir + subdir + "/*.json.gz")
        if len(raw_reviews_filenames) > 0:
            review_filename = raw_reviews_filenames[0].replace(data_dir, "").replace(".json.gz", "")
            print("RUNNING", review_filename)
            runner.run(data_dir, review_filename)
