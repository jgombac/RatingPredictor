import run_configs.parsing_runner as parsing
import run_configs.preprocessing_runner as preprocessing
import run_configs.doc_runner as embedding
import run_configs.tf_runner as tf_runner
import run_configs.simple_runner as simple_runner
import run_configs.result_runner as result_runner


def run(data_dir, reviews_filename):
    # parsing
    parsing.run_configs(data_dir, reviews_filename)

    # preprocessing
    preprocessing.run_configs(data_dir, reviews_filename)

    # simple sentiment
    simple_runner.run_config(data_dir, reviews_filename)

    # doc2vec embedding
    embedding.run_configs(data_dir, reviews_filename)

    # tf training
    tf_runner.run_configs(data_dir, reviews_filename)

    # metrics
    result_runner.run_configs(data_dir, reviews_filename)