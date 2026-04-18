from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os


def train_model(classifier, X_train, y_train):
    model = classifier()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=1),
        "Recall": recall_score(y_test, y_pred, zero_division=1),
        "F1 Score": f1_score(y_test, y_pred, zero_division=1),
    }


def save_model(model, path):
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(model, path)
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