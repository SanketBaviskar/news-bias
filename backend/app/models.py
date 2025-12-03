import os
from tensorflow.keras.models import load_model

def load_bias_model():
    # Construct absolute paths to models
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, 'models')
    
    fair_model_path = os.path.join(models_dir, 'News_Bias_Fair.h5')
    original_model_path = os.path.join(models_dir, 'News_Bias.h5')

    try:
        model = load_model(fair_model_path)
        print(f"Loaded fair model: {fair_model_path}")
        return model
    except Exception as e:
        print(f"Fair model not found or error loading: {e}")
        print(f"Loading original model: {original_model_path}")
        try:
            model = load_model(original_model_path)
            return model
        except Exception as e:
            print(f"Error loading original model: {e}")
            return None
