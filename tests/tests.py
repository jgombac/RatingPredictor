import result_processor as rp

if __name__ == '__main__':
    t1 = "G:\\Downloads\\Development\\RatingPredictor\\data\\tf_models\\books\\reviews_Books_5_reg_lemma_t1_classifier_c1\\confusion_matrix.csv"
    t2 = "G:\\Downloads\\Development\\RatingPredictor\\data\\tf_models\\books\\reviews_Books_5_reg_lemma_t2_classifier_c1\\confusion_matrix.csv"
    t3 = "G:\\Downloads\\Development\\RatingPredictor\\data\\tf_models\\books\\reviews_Books_5_reg_lemma_t3_classifier_c1\\confusion_matrix.csv"
    print(rp.get_metrics(t1))
    print(rp.get_metrics(t2))
    print(rp.get_metrics(t3))

    print("\n")

    t1 = "G:\\Downloads\\Development\\RatingPredictor\\data\\tf_models\\electronics\\reviews_Electronics_5_reg_lemma_t1_classifier_c1\\confusion_matrix.csv"
    t2 = "G:\\Downloads\\Development\\RatingPredictor\\data\\tf_models\\electronics\\reviews_Electronics_5_reg_lemma_t2_classifier_c1\\confusion_matrix.csv"
    t3 = "G:\\Downloads\\Development\\RatingPredictor\\data\\tf_models\\electronics\\reviews_Electronics_5_reg_lemma_t3_classifier_c1\\confusion_matrix.csv"
    print(rp.get_metrics(t1))
    print(rp.get_metrics(t2))
    print(rp.get_metrics(t3))

    print("\n")

    t1 = "G:\\Downloads\\Development\\RatingPredictor\\data\\tf_models\\electronics\\reviews_Electronics_5_reg_lemma_t1_classifier_c2\\confusion_matrix.csv"
    t2 = "G:\\Downloads\\Development\\RatingPredictor\\data\\tf_models\\electronics\\reviews_Electronics_5_reg_lemma_t2_classifier_c2\\confusion_matrix.csv"

    print(rp.get_metrics(t1))
    print(rp.get_metrics(t2))
