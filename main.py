import os
import pandas as pd
from data_loading import load_data
from preprocessing import preprocess_dataframe
from feature_extraction import split_data, extract_features
from embeddings import prepare_word2vec_embeddings, prepare_doc2vec_embeddings, prepare_bert_embeddings
from model_training import train_and_evaluate, save_model
from check_pretrained_models import check_pretrained_model

# Define model file paths
model_files = {
    'count_vectorizer': 'saved_models/RF_count_vectorizer.pkl',
    'tfidf_vectorizer': 'saved_models/RF_tfidf_vectorizer.pkl',
    'word2vec': 'saved_models/RF_word2vec.pkl',
    'doc2vec': 'saved_models/RF_doc2vec.pkl',
    'bert': 'saved_models/RF_bert.pkl'
}

def main():
    df = load_data("data/Spam_Email_Data.csv")
    df = preprocess_dataframe(df)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_counts, X_test_counts, X_train_tfidf, X_test_tfidf = extract_features(X_train, X_test)

    # Prepare embeddings
    X_train_w2v, X_test_w2v = prepare_word2vec_embeddings(X_train, X_test, df)
    X_train_d2v, X_test_d2v = prepare_doc2vec_embeddings(X_train, X_test, df)
    X_train_bert, X_test_bert = prepare_bert_embeddings(X_train, X_test)

    results_counts = results_tfidf = results_w2v = results_d2v = results_bert = []

    for model_type, file_path in model_files.items():
        if os.path.exists(file_path):
            print(f"Pre-trained model found for {model_type}.")
            use_existing = input(f"Do you want to use the existing {model_type} model? (y/n): ").strip().lower()
            if use_existing == 'y':
                model = check_pretrained_model(file_path)
                # Use the model for predictions or evaluation (if needed)
                results = train_and_evaluate(model_type, None, y_train, None, y_test, model)
            else:
                print(f"Training a new model for {model_type}.")
                results = train_and_evaluate(model_type, X_train_counts if model_type == 'count_vectorizer' else
                                                                X_train_tfidf if model_type == 'tfidf_vectorizer' else
                                                                X_train_w2v if model_type == 'word2vec' else
                                                                X_train_d2v if model_type == 'doc2vec' else
                                                                X_train_bert, y_train,
                                                                X_test_counts if model_type == 'count_vectorizer' else
                                                                X_test_tfidf if model_type == 'tfidf_vectorizer' else
                                                                X_test_w2v if model_type == 'word2vec' else
                                                                X_test_d2v if model_type == 'doc2vec' else
                                                                X_test_bert, y_test)
                save_model(results['model'], file_path)  # Save the newly trained model
        else:
            print(f"No pre-trained model found for {model_type}. Training a new model.")
            results = train_and_evaluate(model_type, X_train_counts if model_type == 'count_vectorizer' else
                                                            X_train_tfidf if model_type == 'tfidf_vectorizer' else
                                                            X_train_w2v if model_type == 'word2vec' else
                                                            X_train_d2v if model_type == 'doc2vec' else
                                                            X_train_bert, y_train,
                                                            X_test_counts if model_type == 'count_vectorizer' else
                                                            X_test_tfidf if model_type == 'tfidf_vectorizer' else
                                                            X_test_w2v if model_type == 'word2vec' else
                                                            X_test_d2v if model_type == 'doc2vec' else
                                                            X_test_bert, y_test)
            save_model(results['model'], file_path)  # Save the newly trained model

        # Combine and display results
        all_results = results_counts + results_tfidf + results_w2v + results_d2v + results_bert
        results_df = pd.DataFrame(all_results)
        print(results_df)

if __name__ == "__main__":
    main()
    

