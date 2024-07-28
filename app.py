from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot

app = Flask(__name__)

# Load the saved model
model = load_model('News_Bias.h5')

# Function to preprocess input text
def preprocess_input(text):
    vocab_size = 5000
    sent_length = 20

    # Tokenize and preprocess text
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    onehot_repr = [one_hot(review, vocab_size)]
    embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)

    return np.array(embedded_docs)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']

        # Preprocess the input
        processed_input = preprocess_input(news_text)

        # Make prediction
        prediction = model.predict(processed_input)

        # Interpret prediction
        if prediction[0][0] >= 0.5:
            result = "The news is biased."
        else:
            result = "The news is not biased."

        return render_template('index.html', prediction_text=result, news_text=news_text)

if __name__ == '__main__':
    app.run(debug=True)
