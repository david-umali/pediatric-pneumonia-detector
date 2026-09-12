import os
from pathlib import Path

from flask import Flask, request, render_template

from inference import PredictionService
from preprocessing import preprocess_image


app = Flask(__name__)

# Default to the model directory beside this Python file
base_dir = Path(__file__).resolve().parent
default_model_path = base_dir / "model" / "cnn_best_100.h5"

# Allow the model location to be configured externally
model_path = os.environ.get("MODEL_PATH", str(default_model_path))

# Load the model once per application process
predictor = PredictionService(model_path)

print(
    f"Model input size: "
    f"{predictor.input_width} x {predictor.input_height}"
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


if __name__ == "__main__":
    app.run(debug=True)
