from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)

    CORS(app, resources={r"/api/*": {"origins": "*", "allow_headers": ["Content-Type"]}})

    from app.routes.chat import chat_bp
    from app.routes.health import health_bp

    app.register_blueprint(chat_bp, url_prefix="/api")
    app.register_blueprint(health_bp, url_prefix="/api")

    return app
