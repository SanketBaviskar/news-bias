import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

def preprocess_input(text):
    vocab_size = 5000
    sent_length = 20

    # Tokenize and preprocess text
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in STOPWORDS]
    review = ' '.join(review)
    onehot_repr = [one_hot(review, vocab_size)]
    embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)

    return np.array(embedded_docs)
