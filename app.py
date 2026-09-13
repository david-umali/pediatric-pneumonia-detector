"""Entrypoint for the Flask application."""

from pneumonia_detector import create_app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
