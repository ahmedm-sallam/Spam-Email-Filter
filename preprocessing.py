import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def text_preprocessing(text):
    text = re.sub(r'\S+@\S+', '', text)  # Remove email addresses
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def preprocess_dataframe(df):
    df['text'] = df['text'].apply(text_preprocessing)
    return df