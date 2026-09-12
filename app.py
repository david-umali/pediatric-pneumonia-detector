import os
from pathlib import Path

from flask import Flask, request, render_template

from preprocessing import preprocess_image


def create_app(config=None, predictor=None):
    app = Flask(__name__)

    base_dir = Path(__file__).resolve().parent
    default_model_path = base_dir / "model" / "cnn_best_100.h5"

    app.config.from_mapping(
        MODEL_PATH=os.environ.get("MODEL_PATH", str(default_model_path)),
    )

    # Allow callers, including tests, to override configuration
    if config is not None:
        app.config.update(config)

    # Load the real model only when no predictor was supplied
    if predictor is None:
        from inference import PredictionService

        predictor = PredictionService(app.config["MODEL_PATH"])

    app.logger.info(
        "Model input size: %s x %s",
        predictor.input_width,
        predictor.input_height,
    )

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        if "file" not in request.files or request.files["file"].filename == "":
            return render_template(
                "index.html",
                error="Please select an image file.",
            )

        file = request.files["file"]

        img_array = preprocess_image(
            file.read(),
            width=predictor.input_width,
            height=predictor.input_height,
        )

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
