# Spam Email Filter
 
## Overview
This project focuses on building and comparing several machine-learning models to classify emails as spam or not spam. Utilizing a variety of text processing techniques and classification algorithms, this repository contains all the code and models needed to train, evaluate, and deploy these classifiers.

## Project Structure
- `data/`: Folder containing the dataset used for training and testing the models.
- `models/`: Saved models in `.pkl` format.
- `notebooks/`: Jupyter notebooks detailing the entire process from data preprocessing to model evaluation.
- `vectorizers/`: Saved vectorizer objects to transform raw text data into a suitable format for model prediction.

## Models Used
- Logistic Regression (with CountVectorizer and TF-IDF)
- Random Forest (with CountVectorizer and TF-IDF)
- Logistic Regression and Random Forest using Word2Vec embeddings
- Logistic Regression and Random Forest using Doc2Vec embeddings
- Logistic Regression and Random Forest with BERT embeddings

## Running the Models
To run the saved models on new data, use the following script:
```python
import pickle

def load_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

model = load_model('path/to/your/saved/model.pkl')
# Example: model.predict(new_data)
```

