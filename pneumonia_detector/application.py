import os
import logging
import time
import uuid

from pathlib import Path

from flask import Flask, g, jsonify, render_template, request

from werkzeug.exceptions import RequestEntityTooLarge

from .preprocessing import InvalidImageError, preprocess_image


def create_app(config=None, predictor=None):
    app = Flask(__name__)

    base_dir = Path(__file__).resolve().parent.parent
    default_model_path = base_dir / "model" / "cnn_best_100.h5"

    app.config.from_mapping(
        MODEL_PATH=os.environ.get("MODEL_PATH", str(default_model_path)),
        MODEL_VERSION=os.environ.get("MODEL_VERSION", "cnn_best_100"),
        MAX_CONTENT_LENGTH=int(
            os.environ.get("MAX_CONTENT_LENGTH", 10 * 1024 * 1024)
        ),
        MAX_IMAGE_PIXELS=int(
            os.environ.get("MAX_IMAGE_PIXELS", 16_000_000)
        ),
    )

    if config is not None:
        app.config.update(config)

    app.logger.setLevel(logging.INFO)

    if predictor is None:
        from .inference import PredictionService

        predictor = PredictionService(app.config["MODEL_PATH"])

    app.logger.info(
        "Model input size: %s x %s",
        predictor.input_width,
        predictor.input_height,
    )

    @app.before_request
    def start_request_tracking():
        g.request_id = uuid.uuid4().hex
        g.request_started_at = time.perf_counter()

    @app.after_request
    def log_request(response):
        duration_ms = (
            time.perf_counter() - g.request_started_at
        ) * 1000

        response.headers["X-Request-ID"] = g.request_id

        app.logger.info(
            "request_completed request_id=%s method=%s path=%s "
            "status=%s duration_ms=%.2f",
            g.request_id,
            request.method,
            request.path,
            response.status_code,
            duration_ms,
        )

        return response

    def error_response(code, message, status):
        if request.path.startswith("/api/"):
            return jsonify(
                error={
                    "code": code,
                    "message": message,
                }
            ), status

        return render_template(
            "index.html",
            error=message,
        ), status

    def predict_uploaded_image():
        file = request.files.get("file")

        if file is None or file.filename == "":
            raise InvalidImageError("Please select an image file.")

        img_array = preprocess_image(
            file.read(),
            width=predictor.input_width,
            height=predictor.input_height,
            max_pixels=app.config["MAX_IMAGE_PIXELS"],
        )

        return predictor.predict(img_array)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(error):
        return error_response(
            code="upload_too_large",
            message=(
                "Upload exceeds the request size limit. "
                "Please use a smaller file."
            ),
            status=413,
        )

    @app.errorhandler(InvalidImageError)
    def handle_invalid_image(error):
        return error_response(
            code="invalid_image",
            message=str(error),
            status=400,
        )

    @app.route("/health/live", methods=["GET"])
    def health_live():
        return jsonify(status="alive"), 200

    @app.route("/health/ready", methods=["GET"])
    def health_ready():
        return jsonify(status="ready"), 200

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        result = predict_uploaded_image()

        return render_template(
            "index.html",
            label=result["label"],
            score=round(result["score"], 4),
        )

    @app.route("/api/v1/predict", methods=["POST"])
    def predict_api():
        result = predict_uploaded_image()

        return jsonify(
            label=result["label"],
            score=result["score"],
            model_version=app.config["MODEL_VERSION"],
        )

    return app

