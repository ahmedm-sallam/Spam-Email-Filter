# Spam Email Filter

## Overview
This project focuses on building and comparing several machine-learning models to classify emails as spam or not spam. Utilizing a variety of text processing techniques and classification algorithms, this repository contains all the code and models needed to train, evaluate, and deploy these classifiers.

## Data Preprocessing

The data preprocessing step involves cleaning the email text using various NLP techniques. This includes:

- Removing email addresses, non-alphabetic characters, and HTML tags
- Tokenizing and converting text to lowercase
- Removing stopwords
- Lemmatizing the tokens

## Feature Extraction

### Non-Neural Network-Based Embeddings

We utilize Count Vectorizer and TF-IDF Vectorizer for feature extraction from the cleaned text data.

- **Count Vectorizer**: Converts text into a matrix of token counts.
- **TF-IDF Vectorizer**: Transforms text into a matrix based on term frequency-inverse document frequency.

### Neural Network-Based Embeddings

I employ Word2Vec, Doc2Vec, and BERT models for neural network-based embeddings.

- **Word2Vec**: Creates word embeddings using the skip-gram model.
- **Doc2Vec**: Generates document embeddings.
- **BERT**: Uses pre-trained BERT model to obtain contextual embeddings for the text. BERT embeddings are computed using GPU acceleration for improved performance.

## Training and Evaluation

We train Logistic Regression and Random Forest classifiers on the extracted features. The models are evaluated based on accuracy, precision, recall, and F1 score. Pre-trained models are loaded from the disk if available; otherwise, training is performed and the models are saved for future use. Each model is saved for future use.

## Model Evaluation Results

The table below summarizes the performance metrics of the different models trained using various feature extraction techniques:

| Model                                    | Accuracy | Precision | Recall | F1 Score |
|------------------------------------------|----------|-----------|--------|----------|
| Logistic Regression (count_vectorizer)   | 0.9931   | 0.9904    | 0.9877 | 0.9890   |
| Random Forest (count_vectorizer)         | 0.9858   | 0.9847    | 0.9698 | 0.9772   |
| Logistic Regression (tfidf_vectorizer)   | 0.9763   | 0.9927    | 0.9314 | 0.9611   |
| Random Forest (tfidf_vectorizer)         | 0.9862   | 0.9902    | 0.9657 | 0.9778   |
| Logistic Regression (word2vec)           | 0.9862   | 0.9929    | 0.9630 | 0.9777   |
| Random Forest (word2vec)                 | 0.9836   | 0.9859    | 0.9616 | 0.9736   |
| Logistic Regression (doc2vec)            | 0.9655   | 0.9590    | 0.9300 | 0.9443   |
| Random Forest (doc2vec)                  | 0.9388   | 0.9565    | 0.8436 | 0.8965   |
| Logistic Regression (bert)               | 0.9871   | 0.9834    | 0.9753 | 0.9793   |
| Random Forest (bert)                     | 0.9694   | 0.9910    | 0.9108 | 0.9492   |

## Observations and Analysis

- **Count Vectorizer Models**: High accuracy and precision, indicating good generalization on test data.
- **TF-IDF Vectorizer Models**: High precision, fewer false positives.
- **Word2Vec Models**: Strong performance, particularly in logistic regression.
- **Doc2Vec Models**: Slightly lower performance in recall and F1 score.
- **BERT Models**: Excellent accuracy and precision, with logistic regression outperforming random forest in recall and F1 score.

These results provide valuable insights into the effectiveness of various feature extraction techniques and machine learning models for spam email detection.

## Usage

To run the project:

1. **Install Dependencies**:
   Please make sure you have the required packages installed. You can use the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Main Script**:
   To execute the main script, which will check for pre-trained models and train new ones if needed, run:
   ```bash
   python main.py
   ```

3. **GPU Acceleration**:
   Please make sure your environment is set up to use GPU for BERT embeddings. You need to have CUDA and compatible versions of PyTorch installed.

4. **Check Saved Models**:
   Pre-trained models are saved in the `saved_models` directory. If the models are not available, they will be trained and saved automatically.
