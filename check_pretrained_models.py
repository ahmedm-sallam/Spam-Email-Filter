import os
import joblib


def check_pretrained_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    print(f"Pre-trained model {model_path} not found. Training a new model.")
    return None
