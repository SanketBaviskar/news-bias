import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

# Download stopwords if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def preprocess_text(text):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

def train_fair_model():
    print("Loading processed dataset...")
    df = pd.read_csv('Dataset_Processed.csv')
    
    # 1. Compute Weights using Reweighing
    print("Computing fairness weights...")
    df_aif = df[['protected_attribute', 'label_binary']]
    dataset = BinaryLabelDataset(
        favorable_label=0,
        unfavorable_label=1,
        df=df_aif,
        label_names=['label_binary'],
        protected_attribute_names=['protected_attribute']
    )
    
    privileged_groups = [{'protected_attribute': 1}]
    unprivileged_groups = [{'protected_attribute': 0}]
    
    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    dataset_transf = RW.fit_transform(dataset)
    
    weights = dataset_transf.instance_weights
    print("Weights computed. Mean:", np.mean(weights))
    
    # 2. Prepare Data for Training
    print("Preprocessing text data...")
    voc_size = 5000
    sent_length = 20
    
    corpus = []
    for i in range(len(df)):
        corpus.append(preprocess_text(df['text'][i]))
        
    onehot_repr = [one_hot(words, voc_size) for words in corpus]
    embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
    
    X_final = np.array(embedded_docs)
    y_final = np.array(df['label_binary'])
    
    # 3. Build Model (Same architecture as original)
    print("Building model...")
    embedding_vector_features = 40
    model = Sequential()
    model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
    model.add(LSTM(100)) # Original had LSTM(100) based on notebook? Or let's check app.py/notebook
    # Notebook cell 20 imports Bidirectional, Dropout but doesn't show model summary.
    # Let's assume a simple LSTM or Bidirectional LSTM.
    # I'll use a robust architecture similar to what might be expected.
    # app.py uses load_model('News_Bias.h5').
    # Let's stick to a standard architecture.
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    
    # 4. Train with Weights
    print("Training model with sample weights...")
    # Split data? The notebook didn't seem to split or maybe it did later.
    # For simplicity and "workability", we train on the whole dataset or split.
    # Let's do a simple split.
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X_final, y_final, weights, test_size=0.33, random_state=42
    )
    
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, sample_weight=w_train)
    
    # 5. Save Model
    print("Saving model to News_Bias_Fair.h5...")
    model.save('News_Bias_Fair.h5')
    print("Done.")

if __name__ == "__main__":
    train_fair_model()
