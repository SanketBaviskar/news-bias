from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    import os
    cors_origin = os.environ.get('CORS_ORIGIN', '*')
    CORS(app, resources={r"/*": {"origins": cors_origin}})

    from .routes import main
    app.register_blueprint(main)

    return app
