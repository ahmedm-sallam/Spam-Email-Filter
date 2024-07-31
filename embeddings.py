import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from transformers import BertTokenizer, BertModel
import torch

# Ensure PyTorch uses the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_word2vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1):
    return Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=sg)

def get_word2vec_embedding(text, word2vec_model):
    words = text.split()
    embedding = np.mean([word2vec_model.wv[word] for word in words if word in word2vec_model.wv], axis=0)
    return embedding

def prepare_word2vec_embeddings(X_train, X_test, df):
    sentences = [row.split() for row in df['text']]
    word2vec_model = train_word2vec(sentences)
    X_train_w2v = np.array([get_word2vec_embedding(text, word2vec_model) for text in X_train])
    X_test_w2v = np.array([get_word2vec_embedding(text, word2vec_model) for text in X_test])
    return X_train_w2v, X_test_w2v

def train_doc2vec(tagged_data, vector_size=100, window=5, min_count=1, workers=4, epochs=20):
    return Doc2Vec(tagged_data, vector_size=vector_size, window=window, min_count=min_count, workers=workers, epochs=epochs)

def prepare_doc2vec_embeddings(X_train, X_test, df):
    tagged_data = [TaggedDocument(words=_d.split(), tags=[str(i)]) for i, _d in enumerate(df['text'])]
    doc2vec_model = train_doc2vec(tagged_data)
    X_train_d2v = np.array([doc2vec_model.infer_vector(text.split()) for text in X_train])
    X_test_d2v = np.array([doc2vec_model.infer_vector(text.split()) for text in X_test])
    return X_train_d2v, X_test_d2v

def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()  # Move outputs back to CPU

def prepare_bert_embeddings(X_train, X_test):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)  # Load model to GPU

    X_train_bert = np.array([get_bert_embedding(text, tokenizer, model) for text in X_train])
    X_test_bert = np.array([get_bert_embedding(text, tokenizer, model) for text in X_test])
    return X_train_bert, X_test_bert
