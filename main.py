import os
import pandas as pd
from data_loading import load_data
from preprocessing import preprocess_dataframe
from feature_extraction import split_data, extract_features
from embeddings import prepare_word2vec_embeddings, prepare_doc2vec_embeddings, prepare_bert_embeddings
from model_training import train_and_evaluate, save_model
from check_pretrained_models import check_pretrained_model


def main():
    df = load_data("data/Spam_Email_Data.csv")
    df = preprocess_dataframe(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_counts, X_test_counts, X_train_tfidf, X_test_tfidf = extract_features(X_train, X_test)
    X_train_w2v, X_test_w2v = prepare_word2vec_embeddings(X_train, X_test, df)
    X_train_d2v, X_test_d2v = prepare_doc2vec_embeddings(X_train, X_test, df)
    X_train_bert, X_test_bert = prepare_bert_embeddings(X_train, X_test)

    features = {
        'count_vectorizer': (X_train_counts, X_test_counts),
        'tfidf_vectorizer': (X_train_tfidf, X_test_tfidf),
        'word2vec':         (X_train_w2v,   X_test_w2v),
        'doc2vec':          (X_train_d2v,   X_test_d2v),
        'bert':             (X_train_bert,  X_test_bert),
    }

    all_results = []

    for feature_type, (X_train_feat, X_test_feat) in features.items():
        model_path = f'saved_models/RF_{feature_type}.pkl'

        if os.path.exists(model_path):
            use_existing = input(f"Pre-trained model found for {feature_type}. Use it? (y/n): ").strip().lower()
            if use_existing == 'y':
                print(f"Skipping training for {feature_type}.")
                continue

        results = train_and_evaluate(feature_type, X_train_feat, y_train, X_test_feat, y_test)
        all_results.extend(results)

    if all_results:
        print(pd.DataFrame(all_results))


if __name__ == "__main__":
    main()
