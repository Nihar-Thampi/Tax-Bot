from flask import Flask
from flask_cors import CORS


def create_app():
    app = Flask(__name__)

    CORS(app, resources={r"/api/*": {"origins": "*", "allow_headers": ["Content-Type", "Authorization"]}})

    from app.db import init_db
    init_db()

    from app.routes.auth import auth_bp
    from app.routes.chat import chat_bp
    from app.routes.conversations import conversations_bp
    from app.routes.health import health_bp

    app.register_blueprint(auth_bp, url_prefix="/api")
    app.register_blueprint(chat_bp, url_prefix="/api")
    app.register_blueprint(conversations_bp, url_prefix="/api")
    app.register_blueprint(health_bp, url_prefix="/api")

    return app
