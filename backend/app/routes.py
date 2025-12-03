from flask import Blueprint, request, jsonify
from .models import load_bias_model
from .utils import preprocess_input

main = Blueprint('main', __name__)

# Load model once when the blueprint is registered (or lazily)
model = load_bias_model()

@main.route('/')
def home():
    return "News Bias Detection API Running", 200

@main.route('/health')
def health():
    return "OK", 200

@main.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    if request.method == 'POST':
        data = request.get_json()
        if not data:
             # Fallback for form data if needed, but JSON is preferred
            news_text = request.form.get('news_text')
        else:
            news_text = data.get('news_text')

        if not news_text:
            return jsonify({'error': 'No text provided'}), 400

        # Preprocess the input
        processed_input = preprocess_input(news_text)

        # Make prediction
        prediction = model.predict(processed_input)

        # Interpret prediction
        confidence = float(prediction[0][0])
        if confidence >= 0.5:
            result = "The news is biased."
            explanation = "The model detected patterns often associated with biased reporting."
        else:
            result = "The news is not biased."
            explanation = "The model did not detect significant bias in the text."

        return jsonify({
            'prediction_text': result,
            'confidence': confidence,
            'explanation': explanation
        })
