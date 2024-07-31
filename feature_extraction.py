from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

def split_data(df, test_size=0.4, random_state=135):
    print('Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def extract_features(X_train, X_test):
    print('Extracting features...')
    # Count Vectorizer
    bow_vectorizer = CountVectorizer(ngram_range=(3, 3))
    X_train_counts = bow_vectorizer.fit_transform(X_train)
    X_test_counts = bow_vectorizer.transform(X_test)

    # TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print('Feature extraction completed.')

    return X_train_counts, X_test_counts, X_train_tfidf, X_test_tfidf