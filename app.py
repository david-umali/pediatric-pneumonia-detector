import os
from pathlib import Path

from flask import Flask, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from preprocessing import InvalidImageError, preprocess_image


def create_app(config=None, predictor=None):
    app = Flask(__name__)

    base_dir = Path(__file__).resolve().parent
    default_model_path = base_dir / "model" / "cnn_best_100.h5"

    app.config.from_mapping(
        MODEL_PATH=os.environ.get("MODEL_PATH", str(default_model_path)),
        MAX_CONTENT_LENGTH=int(
            os.environ.get("MAX_CONTENT_LENGTH", 10 * 1024 * 1024)
        ),
        MAX_IMAGE_PIXELS=int(
            os.environ.get("MAX_IMAGE_PIXELS", 16_000_000)
        ),
    )

    if config is not None:
        app.config.update(config)

    if predictor is None:
        from inference import PredictionService

        predictor = PredictionService(app.config["MODEL_PATH"])

    app.logger.info(
        "Model input size: %s x %s",
        predictor.input_width,
        predictor.input_height,
    )

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(error):
        return render_template(
            "index.html",
            error="Upload exceeds the request size limit. Please use a smaller file.",
        ), 413

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        file = request.files.get("file")

        if file is None or file.filename == "":
            return render_template(
                "index.html",
                error="Please select an image file.",
            ), 400

        try:
            img_array = preprocess_image(
                file.read(),
                width=predictor.input_width,
                height=predictor.input_height,
                max_pixels=app.config["MAX_IMAGE_PIXELS"],
            )
        except InvalidImageError as exc:
            return render_template(
                "index.html",
                error=str(exc),
            ), 400

        result = predictor.predict(img_array)

        return render_template(
            "index.html",
            label=result["label"],
            score=round(result["score"], 4),
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
