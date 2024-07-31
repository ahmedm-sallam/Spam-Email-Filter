import os
import pickle


def check_pretrained_model(model_name):
    """
    Checks if a pre-trained model exists for the given name in the saved_models directory.

    Args:
        model_name (str): Name of the pre-trained model (e.g., LR_bert.pkl, RF_tfidf_vectorizer.pkl)

    Returns:
        object or None: The loaded pre-trained model if it exists, otherwise None.
    """

    model_path = os.path.join("saved_models", model_name)
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    else:
        print(f"Pre-trained model {model_name} not found. Training a new model.")
        return None
