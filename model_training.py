from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os


def train_model(classifier, X_train, y_train):
    model = classifier()
    model.fit(X_train, y_train)
    print(f"Model trained: {model}")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)
    f1 = f1_score(y_test, y_pred, zero_division=1)
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    print(f"Model evaluated: {model}")


def save_model(model, filename):
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')

    path = os.path.join('saved_models', filename)

    with open(path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {path}")


def train_and_evaluate(feature_type, X_train, y_train, X_test, y_test):
    print(f"Training and evaluating models using {feature_type} features...")
    results = []

    # Logistic Regression
    lr_model = train_model(LogisticRegression, X_train, y_train)
    lr_results = evaluate_model(lr_model, X_test, y_test)
    lr_results['Model'] = f'Logistic Regression ({feature_type})'
    results.append(lr_results)
    save_model(lr_model, f'LR_{feature_type}.pkl')

    # Random Forest
    rf_model = train_model(RandomForestClassifier, X_train, y_train)
    rf_results = evaluate_model(rf_model, X_test, y_test)
    rf_results['Model'] = f'Random Forest ({feature_type})'
    results.append(rf_results)
    save_model(rf_model, f'RF_{feature_type}.pkl')
    print(f"Training and evaluation completed using {feature_type} features.")

    return results