from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load model once when the application starts
model = load_model("model/cnn_best_100.h5")

# Get model input dimensions dynamically from model
input_height = model.input_shape[1]
input_width  = model.input_shape[2]

print(f"Model input size: {input_width} x {input_height}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files or request.files["file"].filename == "":
        return render_template("index.html", error="Please select an image file.")

    file = request.files["file"]

    # Load and preprocess image
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = img.resize((input_width, input_height))

    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array, verbose=0)
    score = float(prediction[0][0])

    # Classify
    label = "PNEUMONIA" if score >= 0.5 else "NORMAL"

    return render_template("index.html", label=label, score=round(score, 4))


if __name__ == "__main__":
    app.run(debug=True)
